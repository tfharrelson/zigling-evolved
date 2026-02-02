const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
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
            predicted_action: *Tensor(T),
            advantage: *Tensor(T),
            state: *Tensor(T),
        ) TensorError!Tensor(T) {
            // TODO: still getting a feel for the abstraction here admittedly this
            // loss function is a bit weirder than ones that are more typically
            // encountered. So this may seem a little clumsy for now.
            var old_log_probs = try self.old_policy.probs(allocator, state);
            var ratio = try predicted_action.elem_div(allocator, &old_log_probs);
            // TODO: implement the elementwise div function
            const clipped_ratio = try clip(T, allocator, &ratio, self.eps);
            // const unclipped_value = try ratio.elem_mul(allocator, advantage);
            // const clipped_value = try clipped_ratio.elem_mul(allocator, advantage);

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
    var t_shape = [_]usize{2};

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
    var action = try p1.probs(allocator, &t);
    const res = try loss.forward(allocator, &action, &advantage, &t);
    expect(res.shape.len == 1) catch {
        std.debug.print("failed, res shape = {any}\n", .{res.shape});
        return error.TestUnexpectedError;
    };
    expect(res.shape[0] == 2) catch {
        std.debug.print("failed, res shape = {any}\n", .{res.shape});
        return error.TestUnexpectedError;
    };

    expect(res.items[0] > 1.99 and res.items[0] < 2.01) catch {
        std.debug.print("failed, res = {any}\n", .{res});
        return error.TestUnexpectedError;
    };

    // now check whether the epsilon check is working correctly
    loss.eps = 0.1; // with a 2 advantage this should clip to 2.2 (1.1 * 2)
    t.items[0] = 1; // need to set this lower - otherwise probs are essentially 1 and 0 always
    const res2 = try loss.forward(allocator, &action, &advantage, &t);
    expect(res2.items[0] == 2.2) catch {
        std.debug.print("failed, res = {any}\n", .{res2});
        return error.TestUnexpectedError;
    };
}
