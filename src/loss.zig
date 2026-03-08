const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Index = @import("tensor.zig").Index;
const Linear = @import("linear.zig").Linear;
const TensorError = @import("tensor.zig").TensorError;
const Policy = @import("policy.zig").Policy;
const FCN = @import("model.zig").FullyConnectedLayer;
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;

pub fn ClippedSurrogate(comptime T: type) type {
    return struct {
        old_policy: Policy(T),
        eps: T,

        const Self = @This();

        fn forward(
            self: *Self,
            allocator: Allocator,
            predicted_logprobs: *Tensor(T),
            advantage: *Tensor(T),
            state: *Tensor(T),
        ) TensorError!Tensor(T) {
            // TODO: still getting a feel for the abstraction here admittedly this
            // loss function is a bit weirder than ones that are more typically
            // encountered. So this may seem a little clumsy for now.
            var old_log_probs = try self.old_policy.logprobs(allocator, state);
            // TODO: not thrilled with this api - return a pointer to self
            // after these mutation events, which will make them composable
            // a la tensor.subtract(other).exp().log().etc()
            try predicted_logprobs.subtract(&old_log_probs);
            predicted_logprobs.exp();
            const ratio = predicted_logprobs;
            const clipped_ratio = try clip(T, allocator, ratio, self.eps);

            var loss = std.ArrayList(T).initCapacity(allocator, ratio.items.len) catch return TensorError.OutOfMemory;
            var min_item: T = undefined;
            for (ratio.items, clipped_ratio.items) |clip_item, unclip_item| {
                min_item = @min(clip_item, unclip_item);
                loss.append(allocator, min_item) catch return TensorError.OutOfMemory;
            }
            if (advantage.items.len == 1) {
                for (0..loss.items.len) |i| {
                    loss.items[i] *= advantage.items[0];
                }
            } else {
                // batch mode - advantage is a 1d tensor and loss is 2d
                if (ratio.shape.len != 2) {
                    return TensorError.IncompatibleShapeError;
                }
                var loss_index: usize = 0;
                for (0..advantage.items.len) |i| {
                    for (0..ratio.shape[1]) |_| {
                        loss.items[loss_index] *= advantage.items[i];
                        loss_index += 1;
                    }
                }
            }
            return Tensor(T).init(loss.items, ratio.shape);
        }

        pub fn backward(
            self: *Self,
            allocator: Allocator,
            predicted_logprobs: *Tensor(T),
            actions: []usize,
            state: *Tensor(T),
            advantage: *Tensor(T),
        ) TensorError!Tensor(T) {
            // TODO: still getting a feel for the abstraction here admittedly this
            // loss function is a bit weirder than ones that are more typically
            // encountered. So this may seem a little clumsy for now.
            var old_logprobs = try self.old_policy.logprobs(allocator, state);

            // get the acted upon log probs from the old policy now
            // better to just zero out the non-selected log-probs b/c we're gonna need that shape
            // zeroing these out and creating a new tensor is not super ergonomic right now and relies
            // on log_probs being a 2D tensor where the memory is laid out row-wise
            // TODO: come up with a better strategy to avoid the row-wise constraint b/c that may break
            var loss_grad = std.ArrayList(T).initCapacity(allocator, old_logprobs.shape[0]) catch return TensorError.OutOfMemory;
            for (0..old_logprobs.shape[0]) |batch_idx| {
                // TODO: i'm discarding this tensor immediately upon exiting this loop
                // so this memory allocation should be managed by this function
                // this is complicated by the fact that i'd like to support gpu allocation at some point
                // so internal function allocators would have to support both or i need some other abstraction
                var indices = std.ArrayList(Index).initCapacity(allocator, 2) catch return TensorError.OutOfMemory;
                indices.appendSlice(
                    allocator,
                    &[_]Index{ .{ .int = batch_idx }, .{ .all = {} } },
                ) catch return TensorError.OutOfMemory;
                var row = try old_logprobs.get(allocator, indices.items);
                // zero out the unselected items
                for (0..row.items.len) |action_idx| {
                    if (actions[batch_idx] != action_idx) {
                        row.items[action_idx] = 0;
                    } else {
                        // check if ratio is clipped
                        // TODO: improve my allocation strategy here
                        var action_indices = std.ArrayList(Index).initCapacity(allocator, 2) catch return TensorError.OutOfMemory;
                        action_indices.appendSlice(
                            allocator,
                            &[_]Index{ .{ .int = batch_idx }, .{ .int = action_idx } },
                        ) catch return TensorError.OutOfMemory;
                        const predicted_logprob = try predicted_logprobs.get(allocator, action_indices.items);
                        const adv_sign = std.math.sign(advantage.items[batch_idx]);
                        const check = adv_sign * (@exp(predicted_logprob.items[0] - row.items[action_idx]) - 1);
                        if (check > self.eps) {
                            row.items[action_idx] = 0;
                        } else {
                            row.items[action_idx] = @exp(-row.items[action_idx]) * advantage.items[batch_idx];
                        }
                    }
                }
                // this lays out the 2D array in row-wise format
                loss_grad.appendSlice(allocator, row.items) catch return TensorError.OutOfMemory;
            }
            return Tensor(T).init(loss_grad.items, predicted_logprobs.shape);
        }
    };
}

fn clip(comptime T: type, allocator: Allocator, tensor: *Tensor(T), eps: T) TensorError!Tensor(T) {
    var clipped_items = std.ArrayList(T).initCapacity(allocator, tensor.items.len) catch return TensorError.OutOfMemory;
    for (tensor.items) |item| {
        if (item > 1 + eps) {
            clipped_items.append(allocator, 1 + eps) catch return TensorError.OutOfMemory;
        } else if (item < 1 - eps) {
            clipped_items.append(allocator, 1 - eps) catch return TensorError.OutOfMemory;
        } else {
            clipped_items.append(allocator, item) catch return TensorError.OutOfMemory;
        }
    }
    return try Tensor(T).init(clipped_items.items, tensor.shape);
}

test "clipped surrogate happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ 10, -10 };
    var t_shape = [_]usize{ 1, 2 };

    var t = try Tensor(f32).init(&elements, &t_shape);
    var shape = [_]usize{ 2, 2 };

    var l: Linear(f32) = try .init(allocator, &shape, null);
    var new_items = [_]f32{ 2, 0, 0, 1 };
    l.params.items = &new_items;

    var fcn = FCN(f32){ .linear = l };
    const mod = fcn.model();
    var p1: Policy(f32) = try .init(mod, null, null);

    var old_l: Linear(f32) = try .init(allocator, &shape, null);
    var old_items = [_]f32{ 1, 0, 0, 1 };
    old_l.params.items = &old_items;
    var old_fcn = FCN(f32){ .linear = old_l };
    const old_mod = old_fcn.model();
    const p2: Policy(f32) = try .init(old_mod, null, null);

    var loss = ClippedSurrogate(f32){ .old_policy = p2, .eps = 0.5 };

    var adv_elements = [_]f32{2};
    var adv_shape = [_]usize{1};

    var advantage = try Tensor(f32).init(&adv_elements, &adv_shape);
    var pred_logprobs = try p1.logprobs(allocator, &t);
    const res = try loss.forward(allocator, &pred_logprobs, &advantage, &t);
    expect(res.shape.len == 2) catch {
        std.debug.print("failed, res shape len not equal to 1 = {any}\n", .{res.shape});
        return error.TestUnexpectedError;
    };
    expect(res.shape[0] == 1 and res.shape[1] == 2) catch {
        std.debug.print("failed, res shape not equal to 2 = {any}\n", .{res.shape});
        return error.TestUnexpectedError;
    };

    expect(res.items[0] > 1.99 and res.items[0] < 2.01) catch {
        std.debug.print("failed, res = {any}\n", .{res});
        return error.TestUnexpectedError;
    };

    // now check whether the epsilon check is working correctly
    loss.eps = 0.1; // with a 2 advantage this should clip to 2.2 (1.1 * 2)
    t.items[0] = 1; // need to set this lower - otherwise probs are essentially 1 and 0 always
    const res2 = try loss.forward(allocator, &pred_logprobs, &advantage, &t);
    expect(res2.items[0] == 2.2) catch {
        std.debug.print("failed, res first item != 2.2 => {any}\n", .{res2});
        return error.TestUnexpectedError;
    };
}

test "clipped surrogate backwards happy path" {
    const allocator = std.heap.page_allocator;
    var shape = [_]usize{ 2, 2 };

    var old_l: Linear(f32) = try .init(allocator, &shape, null);
    var old_items = [_]f32{ 1, 0, 0, 1 };
    old_l.params.items = &old_items;
    var old_fcn = FCN(f32){ .linear = old_l };
    const old_mod = old_fcn.model();
    const p2: Policy(f32) = try .init(old_mod, null, null);

    var loss = ClippedSurrogate(f32){ .old_policy = p2, .eps = 0.5 };

    var new_items = [_]f32{ @log(0.75), @log(0.25), @log(0.1), @log(0.9) };
    var pred_logprobs = try Tensor(f32).init(&new_items, &shape);

    var action_items = [_]usize{ 0, 1 };

    var elements = [_]f32{ 10, -10, 0.1, -0.1 };
    var state_shape = [_]usize{ 2, 2 };
    var state = try Tensor(f32).init(&elements, &state_shape);

    var adv_shape = [_]usize{2};
    var advantage_items = [_]f32{ 0.5, 0.25 };
    var advantages = try Tensor(f32).init(&advantage_items, &adv_shape);

    const grad = try loss.backward(allocator, &pred_logprobs, &action_items, &state, &advantages);
    // NOTE: only the first item in the first row should be non-zero
    // it's value should be very close to 0.5 b/c the advantage is 0.5
    // and the old policy probability of that action is approaching 1.0
    // this means the gradient is 0.5 * 1 / 0.999999 which is a bit higher
    // than 0.5 but not by much
    try expect(grad.items[0] > 0.5 and grad.items[0] < 0.51);
    for (1..grad.items.len) |i| {
        try expect(grad.items[i] == 0);
    }
}
