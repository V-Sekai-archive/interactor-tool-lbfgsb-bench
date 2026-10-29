// vec_vk -- a host Vulkan backend for lbv::Vec.
//
// The guest dispatches the Lean-emitted SPIR-V through rd_compute (Godot's
// RenderingDevice over the sandbox boundary). On a host there is no Godot, so
// this backend talks to Vulkan directly: same SPIR-V kernels, same bindings,
// same params words, same one-submit-per-phase recording, but through the
// loader (MoltenVK on macOS, the real driver on Linux).
//
// Buffers are host-visible and coherent and every phase ends with a fence, so
// upload/read are plain memcpys. That is not the fastest possible setup (a
// device-local buffer with staging copies would be), but it isolates the
// kernels: the number this backend produces is the driver's GPU cost without
// a transfer pipeline on top of it.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <string>

#ifndef LBV_VEC_H
#define LBV_VEC_H ".deps/contract-lbfgsb/guest/drape/lbfgsb_vec.h"
#endif
#include LBV_VEC_H

class VecVk : public lbv::Vec {
public:
	// spv_dir holds one <kernel>.spv per kernel (kernels/drape/gen.sh's
	// spirv step). Returns false and sets err when no device or no kernels.
	static bool probe(std::string &err);
	VecVk();
	~VecVk() override;

	bool open(const std::string &spv_dir, std::string &err);
	const char *name() const override { return "vk"; }
	bool setup(uint32_t n, uint32_t mcap, std::string &err) override;
	bool upload(lbv::Buf b, const void *data, size_t words, size_t off = 0) override;
	bool run(const std::vector<lbv::Op> &ops) override;
	bool pending() const override { return false; }
	void sync() override {}
	bool read(lbv::Buf b, void *out, size_t words, size_t off = 0) override;
	const std::string &error() const override { return err_; }
	// The device's name, for the report.
	const std::string &device_name() const { return dev_name_; }
	// Mean cost of an empty submit + fence, the floor under every phase.
	double empty_submit_us(int reps = 50);
	// Descriptor sets built so far (the cache keeps later dispatches off this path).
	int64_t sets_built() const;

private:
	struct Impl;
	Impl *p_ = nullptr;
	std::string err_;
	std::string dev_name_;
};
