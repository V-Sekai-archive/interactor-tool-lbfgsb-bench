// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "vec_vk.h"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

// One dispatch as the GPU sees it: kernel, one buffer per binding slot, the
// params words for binding 0's constant buffer, and the workgroup count. The
// same description vec_rd.cpp builds for rd_compute.
struct Dispatch {
	const char *kernel = nullptr;
	std::vector<int> binds;  // binding slot -> lbv::Buf (slot 0 is the params CB)
	std::vector<uint32_t> params;
	uint32_t groups = 1;
};

uint32_t fbits(float f) {
	uint32_t u;
	std::memcpy(&u, &f, 4);
	return u;
}

Dispatch describe(const lbv::Op &o, uint32_t n, uint32_t mc) {
	using lbv::Op;
	Dispatch d;
	const uint32_t perElem = (n + 255) / 256;
	switch (o.kind) {
		case Op::BoxProject:
			d.kernel = "lb_box_project";
			d.binds = { o.a, lbv::LB, lbv::UB };
			d.params = { n };
			d.groups = perElem;
			break;
		case Op::PgInf:
			d.kernel = "lb_pg_inf";
			d.binds = { o.a, o.b, lbv::LB, lbv::UB, lbv::SC };
			d.params = { n, o.u0 };
			d.groups = 1;
			break;
		case Op::StepMax:
			d.kernel = "lb_step_max";
			d.binds = { o.a, o.b, lbv::LB, lbv::UB, lbv::SC };
			d.params = { n, o.u0 };
			d.groups = 1;
			break;
		case Op::MultiDot:
			d.kernel = "lb_multi_dot";
			d.binds = { o.a, o.b, o.c };
			d.params = { n, o.u0, 0u, o.u1 ? o.u1 : n, 0u, o.u3 };
			d.groups = o.u0;
			break;
		case Op::Compact:
			d.kernel = "lb_compact";
			d.binds = { lbv::DOTS, lbv::ST, lbv::BF };
			d.params = { mc, o.u0 };
			break;
		case Op::RingStore:
			d.kernel = "lb_ring_store";
			d.binds = { lbv::ST, lbv::SV, lbv::YV, lbv::SM, lbv::YM };
			d.params = { n };
			d.groups = perElem;
			break;
		case Op::ApplyM:
			d.kernel = "lb_apply_m";
			d.binds = { lbv::ST, lbv::BF, lbv::VIN, lbv::VOUT };
			d.params = { mc, o.u0, o.u1 };
			break;
		case Op::Cauchy:
			d.kernel = "lb_cauchy";
			d.binds = { lbv::X, lbv::G, lbv::LB, lbv::UB, lbv::SM, lbv::YM, lbv::ST, lbv::BF,
						lbv::XCP, lbv::VECC, lbv::WK, lbv::ISET };
			d.params = { n, mc };
			break;
		case Op::Subspace:
			d.kernel = "lb_subspace";
			d.binds = { lbv::X, lbv::G, lbv::LB, lbv::UB, lbv::SM, lbv::YM, lbv::ST, lbv::BF,
						lbv::XCP, lbv::VECC, lbv::ISET, lbv::D, lbv::WK };
			d.params = { n, mc, o.u0 };
			break;
		case Op::Saxpby:
			d.kernel = "saxpby";
			d.binds = { o.a, o.b, o.c };
			d.params = { n, fbits(o.fa), fbits(o.fb) };
			d.groups = perElem;
			break;
	}
	return d;
}

const char *const kKernelNames[] = { "lb_box_project", "lb_pg_inf",     "lb_step_max", "lb_multi_dot", "lb_compact",
									 "lb_ring_store",  "lb_apply_m",    "lb_cauchy",    "lb_subspace",  "saxpby" };

std::vector<char> slurp(const std::string &p) {
	std::ifstream in(p, std::ios::binary);
	return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool has_ext(const std::vector<VkExtensionProperties> &v, const char *name) {
	for (const auto &e : v) {
		if (std::strcmp(e.extensionName, name) == 0) return true;
	}
	return false;
}

// 64 bytes per params block: the largest kernel struct is 24 bytes and a
// uniform buffer offset must be a multiple of 16.
constexpr uint32_t kParamSlot = 64;
constexpr uint32_t kMaxOpsPerPhase = 512;

} // namespace

struct VecVk::Impl {
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkDevice dev = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t qfam = 0;
	VkCommandPool cmd_pool = VK_NULL_HANDLE;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	VkFence fence = VK_NULL_HANDLE;
	VkDescriptorPool desc_pool = VK_NULL_HANDLE;

	struct Kernel {
		VkShaderModule mod = VK_NULL_HANDLE;
		VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
		VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
		VkPipeline pipe = VK_NULL_HANDLE;
		uint32_t n_binds = 0;  // storage buffers after the params CB
		double us = 0.0;       // LBFGSB_VK_PERDISPATCH=1 diagnostic
		int64_t calls = 0;
	};
	Kernel kernels[sizeof(kKernelNames) / sizeof(kKernelNames[0])];

	struct Buf {
		VkBuffer buf = VK_NULL_HANDLE;
		VkDeviceMemory mem = VK_NULL_HANDLE;
		void *map = nullptr;
		size_t bytes = 0;
	};
	Buf bufs[lbv::kNumBufs];
	Buf params;
	uint32_t param_slot = kParamSlot;  // max(kParamSlot, minUniformBufferOffsetAlignment)

	// A descriptor set depends only on (kernel, the storage buffers it binds):
	// the params block is a dynamic uniform offset, so a set that has been
	// built once can be rebound for every later op with the same signature.
	// Without this cache each dispatch pays an allocate + update, which on
	// MoltenVK costs more than the dispatch itself.
	std::map<std::string, VkDescriptorSet> set_cache;
	int64_t sets_built = 0;

	VkBuffer make_buffer(size_t bytes, VkBufferUsageFlags usage, VkDeviceMemory *mem_out) {
		VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bi.size = bytes;
		bi.usage = usage;
		bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VkBuffer b = VK_NULL_HANDLE;
		if (vkCreateBuffer(dev, &bi, nullptr, &b) != VK_SUCCESS) return VK_NULL_HANDLE;
		VkMemoryRequirements req{};
		vkGetBufferMemoryRequirements(dev, b, &req);
		VkPhysicalDeviceMemoryProperties mp{};
		vkGetPhysicalDeviceMemoryProperties(phys, &mp);
		uint32_t idx = UINT32_MAX;
		for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
			const bool host = (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
			const bool coh = (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
			if ((req.memoryTypeBits & (1u << i)) && host && coh) {
				idx = i;
				break;
			}
		}
		if (idx == UINT32_MAX) return VK_NULL_HANDLE;
		VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
		ai.allocationSize = req.size;
		ai.memoryTypeIndex = idx;
		VkDeviceMemory m = VK_NULL_HANDLE;
		if (vkAllocateMemory(dev, &ai, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
		if (vkBindBufferMemory(dev, b, m, 0) != VK_SUCCESS) return VK_NULL_HANDLE;
		*mem_out = m;
		return b;
	}
};

VecVk::VecVk() :
		p_(new Impl) {
}

VecVk::~VecVk() {
	if (!p_) return;
	Impl &d = *p_;
	{
		bool any = false;
		for (auto &k : d.kernels) {
			if (k.calls) any = true;
		}
		if (any) {
			std::fprintf(stderr, "# per-dispatch diagnostic (submit+fence after every dispatch)\n");
			for (size_t i = 0; i < sizeof(kKernelNames) / sizeof(kKernelNames[0]); ++i) {
				const Impl::Kernel &k = d.kernels[i];
				if (!k.calls) continue;
				std::fprintf(stderr, "#   %-16s %6lld calls  %9.1f us total  %8.2f us/dispatch\n", kKernelNames[i],
							 (long long)k.calls, k.us, k.us / double(k.calls));
			}
		}
	}
	if (d.dev != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(d.dev);
		for (auto &b : d.bufs) {
			if (b.buf) vkDestroyBuffer(d.dev, b.buf, nullptr);
			if (b.mem) vkFreeMemory(d.dev, b.mem, nullptr);
		}
		if (d.params.buf) vkDestroyBuffer(d.dev, d.params.buf, nullptr);
		if (d.params.mem) vkFreeMemory(d.dev, d.params.mem, nullptr);
		for (auto &k : d.kernels) {
			if (k.pipe) vkDestroyPipeline(d.dev, k.pipe, nullptr);
			if (k.pipe_layout) vkDestroyPipelineLayout(d.dev, k.pipe_layout, nullptr);
			if (k.set_layout) vkDestroyDescriptorSetLayout(d.dev, k.set_layout, nullptr);
			if (k.mod) vkDestroyShaderModule(d.dev, k.mod, nullptr);
		}
		if (d.desc_pool) vkDestroyDescriptorPool(d.dev, d.desc_pool, nullptr);
		if (d.cmd_pool) vkDestroyCommandPool(d.dev, d.cmd_pool, nullptr);
		vkDestroyDevice(d.dev, nullptr);
	}
	if (d.instance) vkDestroyInstance(d.instance, nullptr);
	delete p_;
	p_ = nullptr;
}

bool VecVk::probe(std::string &err) {
	VecVk v;
	const bool ok = v.open(".", err);
	return ok;
}

bool VecVk::open(const std::string &spv_dir, std::string &err) {
	Impl &d = *p_;
	uint32_t n_ext = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &n_ext, nullptr);
	std::vector<VkExtensionProperties> exts(n_ext);
	vkEnumerateInstanceExtensionProperties(nullptr, &n_ext, exts.data());
	const bool portability = has_ext(exts, "VK_KHR_portability_enumeration");

	VkApplicationInfo ai{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
	ai.pApplicationName = "lbfgsb-bench";
	ai.apiVersion = VK_API_VERSION_1_1;
	VkInstanceCreateInfo ci{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	ci.pApplicationInfo = &ai;
	std::vector<const char *> layers;
	std::vector<const char *> iexts;
	if (portability) {
		iexts.push_back("VK_KHR_portability_enumeration");
		ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	}
	ci.enabledExtensionCount = uint32_t(iexts.size());
	ci.ppEnabledExtensionNames = iexts.data();
	ci.enabledLayerCount = 0;
	if (vkCreateInstance(&ci, nullptr, &d.instance) != VK_SUCCESS) {
		err = "vkCreateInstance failed (is a Vulkan loader present?)";
		return false;
	}

	uint32_t n_dev = 0;
	vkEnumeratePhysicalDevices(d.instance, &n_dev, nullptr);
	if (n_dev == 0) {
		err = "no Vulkan physical device";
		return false;
	}
	std::vector<VkPhysicalDevice> devs(n_dev);
	vkEnumeratePhysicalDevices(d.instance, &n_dev, devs.data());
	d.phys = devs[0];
	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(d.phys, &props);
	dev_name_ = props.deviceName;
	d.param_slot = props.limits.minUniformBufferOffsetAlignment > kParamSlot
						   ? uint32_t(props.limits.minUniformBufferOffsetAlignment)
						   : kParamSlot;
	std::fprintf(stderr, "# vk: %s, minUniformBufferOffsetAlignment %u, param slot %u\n", props.deviceName,
				 props.limits.minUniformBufferOffsetAlignment, d.param_slot);

	uint32_t n_qf = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(d.phys, &n_qf, nullptr);
	std::vector<VkQueueFamilyProperties> qf(n_qf);
	vkGetPhysicalDeviceQueueFamilyProperties(d.phys, &n_qf, qf.data());
	bool found = false;
	for (uint32_t i = 0; i < n_qf; ++i) {
		if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			d.qfam = i;
			found = true;
			break;
		}
	}
	if (!found) {
		err = "no compute queue family";
		return false;
	}

	// MoltenVK exposes VK_KHR_portability_subset, which must be enabled.
	uint32_t n_dext = 0;
	vkEnumerateDeviceExtensionProperties(d.phys, nullptr, &n_dext, nullptr);
	std::vector<VkExtensionProperties> dexts(n_dext);
	vkEnumerateDeviceExtensionProperties(d.phys, nullptr, &n_dext, dexts.data());
	std::vector<const char *> dext_names;
	if (has_ext(dexts, "VK_KHR_portability_subset")) dext_names.push_back("VK_KHR_portability_subset");

	const float prio = 1.0f;
	VkDeviceQueueCreateInfo qi{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	qi.queueFamilyIndex = d.qfam;
	qi.queueCount = 1;
	qi.pQueuePriorities = &prio;
	VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qi;
	dci.enabledExtensionCount = uint32_t(dext_names.size());
	dci.ppEnabledExtensionNames = dext_names.data();
	if (vkCreateDevice(d.phys, &dci, nullptr, &d.dev) != VK_SUCCESS) {
		err = "vkCreateDevice failed";
		return false;
	}
	vkGetDeviceQueue(d.dev, d.qfam, 0, &d.queue);

	VkCommandPoolCreateInfo cpi{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	cpi.queueFamilyIndex = d.qfam;
	if (vkCreateCommandPool(d.dev, &cpi, nullptr, &d.cmd_pool) != VK_SUCCESS) {
		err = "vkCreateCommandPool failed";
		return false;
	}
	VkCommandBufferAllocateInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	cbi.commandPool = d.cmd_pool;
	cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbi.commandBufferCount = 1;
	if (vkAllocateCommandBuffers(d.dev, &cbi, &d.cmd) != VK_SUCCESS) {
		err = "vkAllocateCommandBuffers failed";
		return false;
	}
	VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	if (vkCreateFence(d.dev, &fi, nullptr, &d.fence) != VK_SUCCESS) {
		err = "vkCreateFence failed";
		return false;
	}

	VkDescriptorPoolSize sizes[2] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4096 },
									  { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096 * 16 } };
	VkDescriptorPoolCreateInfo dpi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
	dpi.maxSets = 4096;
	dpi.poolSizeCount = 2;
	dpi.pPoolSizes = sizes;
	if (vkCreateDescriptorPool(d.dev, &dpi, nullptr, &d.desc_pool) != VK_SUCCESS) {
		err = "vkCreateDescriptorPool failed";
		return false;
	}

	// The kernels.
	for (size_t i = 0; i < sizeof(kKernelNames) / sizeof(kKernelNames[0]); ++i) {
		const std::string path = spv_dir + "/" + kKernelNames[i] + ".spv";
		std::vector<char> code = slurp(path);
		if (code.empty() || code.size() % 4 != 0) {
			err = "cannot read " + path + " (run the spirv step first)";
			return false;
		}
		VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		smi.codeSize = code.size();
		smi.pCode = reinterpret_cast<const uint32_t *>(code.data());
		Impl::Kernel &k = d.kernels[i];
		if (vkCreateShaderModule(d.dev, &smi, nullptr, &k.mod) != VK_SUCCESS) {
			err = std::string("vkCreateShaderModule failed for ") + kKernelNames[i];
			return false;
		}
	}
	// Binding counts come from the emitted Slang declarations, which vec_rd's
	// table encodes: params plus the kernel's storage buffers.
	const uint32_t nb[sizeof(kKernelNames) / sizeof(kKernelNames[0])] = { 3, 5, 5, 3, 3, 5, 4, 12, 13, 3 };
	for (size_t i = 0; i < sizeof(kKernelNames) / sizeof(kKernelNames[0]); ++i) {
		Impl::Kernel &k = d.kernels[i];
		k.n_binds = nb[i];
		std::vector<VkDescriptorSetLayoutBinding> b(k.n_binds + 1);
		for (uint32_t s = 0; s <= k.n_binds; ++s) {
			b[s].binding = s;
			b[s].descriptorType =
					s == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			b[s].descriptorCount = 1;
			b[s].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		VkDescriptorSetLayoutCreateInfo sli{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		sli.bindingCount = uint32_t(b.size());
		sli.pBindings = b.data();
		if (vkCreateDescriptorSetLayout(d.dev, &sli, nullptr, &k.set_layout) != VK_SUCCESS) {
			err = std::string("descriptor set layout failed for ") + kKernelNames[i];
			return false;
		}
		VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
		pli.setLayoutCount = 1;
		pli.pSetLayouts = &k.set_layout;
		if (vkCreatePipelineLayout(d.dev, &pli, nullptr, &k.pipe_layout) != VK_SUCCESS) {
			err = "pipeline layout failed";
			return false;
		}
		VkComputePipelineCreateInfo pci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		pci.stage.module = k.mod;
		pci.stage.pName = "main";
		pci.layout = k.pipe_layout;
		if (vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &pci, nullptr, &k.pipe) != VK_SUCCESS) {
			err = std::string("compute pipeline failed for ") + kKernelNames[i];
			return false;
		}
	}
	return true;
}

bool VecVk::setup(uint32_t n, uint32_t mcap, std::string &err) {
	Impl &d = *p_;
	if (d.dev == VK_NULL_HANDLE) {
		err = "device not open";
		return false;
	}
	vkDeviceWaitIdle(d.dev);
	// Repetitions call setup() again with the same shape: keep the buffers and
	// just zero them, so a solve's timing does not carry an allocation.
	const bool reuse = (n_ == n && mc_ == mcap && d.bufs[0].buf != VK_NULL_HANDLE);
	if (!reuse) {
		// Every cached descriptor set names the VkBuffers about to be
		// destroyed. Keeping them would leave the kernels writing into freed
		// memory and the driver reading zeros from the live buffers.
		vkResetDescriptorPool(d.dev, d.desc_pool, 0);
		d.set_cache.clear();
	}
	for (int i = 0; i < lbv::kNumBufs; ++i) {
		Impl::Buf &b = d.bufs[i];
		if (b.buf) {
			if (reuse) {
				std::memset(b.map, 0, b.bytes);
				continue;
			}
			vkDestroyBuffer(d.dev, b.buf, nullptr);
			vkFreeMemory(d.dev, b.mem, nullptr);
			b = Impl::Buf{};
		}
		const size_t bytes = lbv::buf_words(lbv::Buf(i), n, mcap) * sizeof(uint32_t);
		b.buf = d.make_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
									   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
							  &b.mem);
		if (!b.buf) {
			err = "buffer allocation failed";
			return false;
		}
		b.bytes = bytes;
		if (vkMapMemory(d.dev, b.mem, 0, bytes, 0, &b.map) != VK_SUCCESS) {
			err = "vkMapMemory failed";
			return false;
		}
		std::memset(b.map, 0, bytes);
	}
	if (!d.params.buf) {
		const size_t bytes = size_t(d.param_slot) * kMaxOpsPerPhase;
		d.params.buf = d.make_buffer(bytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &d.params.mem);
		if (!d.params.buf) {
			err = "params buffer allocation failed";
			return false;
		}
		d.params.bytes = bytes;
		if (vkMapMemory(d.dev, d.params.mem, 0, bytes, 0, &d.params.map) != VK_SUCCESS) {
			err = "vkMapMemory(params) failed";
			return false;
		}
	}
	n_ = n;
	mc_ = mcap;
	return true;
}

double VecVk::empty_submit_us(int reps) {
	Impl &d = *p_;
	if (d.dev == VK_NULL_HANDLE) return 0.0;
	vkResetCommandBuffer(d.cmd, 0);
	VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(d.cmd, &bi);
	vkEndCommandBuffer(d.cmd);
	VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &d.cmd;
	const auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i) {
		vkQueueSubmit(d.queue, 1, &si, d.fence);
		vkWaitForFences(d.dev, 1, &d.fence, VK_TRUE, UINT64_MAX);
		vkResetFences(d.dev, 1, &d.fence);
	}
	const auto t1 = std::chrono::steady_clock::now();
	return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

int64_t VecVk::sets_built() const {
	return p_ ? p_->sets_built : 0;
}

bool VecVk::upload(lbv::Buf b, const void *data, size_t words, size_t off) {
	Impl &d = *p_;
	if (b < 0 || b >= lbv::kNumBufs || !d.bufs[b].map) {
		err_ = "upload: bad buffer";
		return false;
	}
	if ((off + words) * 4 > d.bufs[b].bytes) {
		err_ = "upload: out of range";
		return false;
	}
	std::memcpy(static_cast<char *>(d.bufs[b].map) + off * 4, data, words * 4);
	return true;
}

bool VecVk::read(lbv::Buf b, void *out, size_t words, size_t off) {
	Impl &d = *p_;
	if (b < 0 || b >= lbv::kNumBufs || !d.bufs[b].map) {
		err_ = "read: bad buffer";
		return false;
	}
	if ((off + words) * 4 > d.bufs[b].bytes) {
		err_ = "read: out of range";
		return false;
	}
	std::memcpy(out, static_cast<const char *>(d.bufs[b].map) + off * 4, words * 4);
	return true;
}

bool VecVk::run(const std::vector<lbv::Op> &ops) {
	Impl &d = *p_;
	if (ops.empty()) return true;
	if (ops.size() > kMaxOpsPerPhase) {
		err_ = "phase has more ops than the params arena holds";
		return false;
	}
	// Params for every op go into the arena before the single submit.
	for (size_t i = 0; i < ops.size(); ++i) {
		const Dispatch disp = describe(ops[i], n_, mc_);
		uint32_t *slot = reinterpret_cast<uint32_t *>(static_cast<char *>(d.params.map) + i * d.param_slot);
		std::memset(slot, 0, d.param_slot);
		std::memcpy(slot, disp.params.data(), disp.params.size() * 4);
	}

	vkResetCommandBuffer(d.cmd, 0);
	VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (vkBeginCommandBuffer(d.cmd, &bi) != VK_SUCCESS) {
		err_ = "vkBeginCommandBuffer failed";
		return false;
	}
	// LBFGSB_VK_PERDISPATCH=1 submits and waits after every dispatch and
	// accumulates the time per kernel: a diagnostic for where a phase's time
	// goes, not the benchmark's own path.
	const bool per_dispatch = std::getenv("LBFGSB_VK_PERDISPATCH") != nullptr;

	int64_t dispatches = 0;
	for (size_t i = 0; i < ops.size(); ++i) {
		const Dispatch disp = describe(ops[i], n_, mc_);
		int ki = -1;
		for (size_t k = 0; k < sizeof(kKernelNames) / sizeof(kKernelNames[0]); ++k) {
			if (std::strcmp(kKernelNames[k], disp.kernel) == 0) {
				ki = int(k);
				break;
			}
		}
		if (ki < 0) {
			err_ = std::string("no pipeline for ") + disp.kernel;
			return false;
		}
		Impl::Kernel &k = d.kernels[ki];

		// One set per (kernel, buffers): built on first use, then rebound with
		// the params block's dynamic offset.
		std::string key = disp.kernel;
		for (int b : disp.binds) {
			key += ":" + std::to_string(b);
		}
		VkDescriptorSet set = VK_NULL_HANDLE;
		auto it = d.set_cache.find(key);
		if (it != d.set_cache.end()) {
			set = it->second;
		} else {
			VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			ai.descriptorPool = d.desc_pool;
			ai.descriptorSetCount = 1;
			ai.pSetLayouts = &k.set_layout;
			if (vkAllocateDescriptorSets(d.dev, &ai, &set) != VK_SUCCESS) {
				err_ = "vkAllocateDescriptorSets failed";
				return false;
			}
			std::vector<VkDescriptorBufferInfo> infos(k.n_binds + 1);
			std::vector<VkWriteDescriptorSet> writes(k.n_binds + 1);
			infos[0].buffer = d.params.buf;
			infos[0].offset = 0;  // the offset is dynamic, supplied at bind time
			infos[0].range = d.param_slot;
			for (uint32_t s = 1; s <= k.n_binds; ++s) {
				const int buf = disp.binds[s - 1];
				if (buf < 0 || buf >= lbv::kNumBufs || !d.bufs[buf].buf) {
					err_ = "dispatch binds an unallocated buffer";
					return false;
				}
				infos[s].buffer = d.bufs[buf].buf;
				infos[s].offset = 0;
				infos[s].range = VK_WHOLE_SIZE;
			}
			for (uint32_t s = 0; s <= k.n_binds; ++s) {
				writes[s].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[s].dstSet = set;
				writes[s].dstBinding = s;
				writes[s].descriptorCount = 1;
				writes[s].descriptorType =
						s == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				writes[s].pBufferInfo = &infos[s];
			}
			vkUpdateDescriptorSets(d.dev, uint32_t(writes.size()), writes.data(), 0, nullptr);
			d.set_cache[key] = set;
			++d.sets_built;
		}

		const uint32_t dyn = uint32_t(i) * d.param_slot;
		vkCmdBindPipeline(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
		vkCmdBindDescriptorSets(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe_layout, 0, 1, &set, 1, &dyn);
		vkCmdDispatch(d.cmd, disp.groups, 1, 1);
		++dispatches;
		if (per_dispatch) {
			// End, submit and wait here so this dispatch's own cost is the
			// delta; then start a fresh command buffer for the rest.
			if (vkEndCommandBuffer(d.cmd) != VK_SUCCESS) {
				err_ = "vkEndCommandBuffer failed";
				return false;
			}
			VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
			si.commandBufferCount = 1;
			si.pCommandBuffers = &d.cmd;
			const auto t0 = std::chrono::steady_clock::now();
			if (vkQueueSubmit(d.queue, 1, &si, d.fence) != VK_SUCCESS) {
				err_ = "vkQueueSubmit failed";
				return false;
			}
			if (vkWaitForFences(d.dev, 1, &d.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
				err_ = "vkWaitForFences failed";
				return false;
			}
			vkResetFences(d.dev, 1, &d.fence);
			const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
			Impl::Kernel &kk = d.kernels[ki];
			kk.us += us;
			++kk.calls;
			if (vkResetCommandBuffer(d.cmd, 0) != VK_SUCCESS || vkBeginCommandBuffer(d.cmd, &bi) != VK_SUCCESS) {
				err_ = "command buffer restart failed";
				return false;
			}
		}
	}
	if (vkEndCommandBuffer(d.cmd) != VK_SUCCESS) {
		err_ = "vkEndCommandBuffer failed";
		return false;
	}
	VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &d.cmd;
	if (vkQueueSubmit(d.queue, 1, &si, d.fence) != VK_SUCCESS) {
		err_ = "vkQueueSubmit failed";
		return false;
	}
	if (vkWaitForFences(d.dev, 1, &d.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
		err_ = "vkWaitForFences failed";
		return false;
	}
	vkResetFences(d.dev, 1, &d.fence);
	++phases_;
	dispatches_ += dispatches;
	return true;
}
