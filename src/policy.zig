const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const Model = @import("model.zig").Model;
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("tensor.zig").TensorError;
const Linear = @import("linear.zig").Linear;
const FCN = @import("model.zig").FullyConnectedLayer;
const Update = @import("model.zig").Update;

pub fn Policy(comptime T: type) type {
    return struct {
        const Self = @This();
        model: Model(T),
        // model: *FCN(T),
        rand: std.Random,
        // TODO: implement an options pattern complete with nice defaults
        eps: T,

        pub fn init(model: Model(T), eps: ?T, seed: ?usize) TensorError!Self {
            var random_seed: usize = undefined;
            if (seed) |s| {
                random_seed = s;
            } else {
                std.posix.getrandom(std.mem.asBytes(&random_seed)) catch return TensorError.UnexpectedError;
            }
            var pnrg = std.Random.DefaultPrng.init(random_seed);
            const rand = pnrg.random();

            var epsilon: T = undefined;
            if (eps) |e| {
                epsilon = e;
            } else {
                epsilon = rand.float(T);
            }
            return .{ .model = model, .rand = rand, .eps = epsilon };
        }

        pub fn forward(self: *Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
            return self.model.forward(alloc, tensor);
        }

        pub fn backward(self: *Self, alloc: Allocator, tensor: *Tensor(T), update: ?Update(T)) TensorError!Tensor(T) {
            return self.model.backward(alloc, tensor, update);
        }

        pub fn probs(self: *Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
            var logits = try self.forward(alloc, tensor);
            return try Softmax(T, alloc, &logits);
        }

        pub fn logprobs(self: *Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
            var probas = self.probs(alloc, tensor);
            probas.log(); // cast probs to log probs in place
            return probas;
        }

        pub fn greedy_action(self: *Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!usize {
            // NOTE: this will call forward and find the best action
            // Should I assume the output is log-prob instead of likelihoods?
            // in which case i should convert to positive definite values with the exp func
            var distribution = try self.forward(alloc, tensor);
            return distribution.argmax();
        }

        pub fn sample_action(self: *Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!usize {
            const distribution = try self.forward(alloc, tensor);
            // normalize the distribution by finding its sum
            // Should I assume the output is log-prob instead of likelihoods?
            // in which case i should convert to positive definite values with the exp func
            var dist_sum: T = 0;
            for (distribution.items) |item| {
                dist_sum += item;
            }
            // draw a random number
            const rand_value = self.rand.float(T);
            // hopefully this works for all T...
            var curr_prob: T = 0;
            for (0..distribution.items.len) |idx| {
                if (rand_value < curr_prob + (distribution.items[idx] / dist_sum)) {
                    return idx;
                }
                curr_prob += distribution.items[idx] / dist_sum;
            }
            unreachable;
        }

        pub fn epsilon_action(self: *Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!usize {
            const rand_value = self.rand.float(T);
            if (rand_value < self.eps) {
                return self.sample_action(alloc, tensor);
            } else {
                return self.greedy_action(alloc, tensor);
            }
        }
    };
}

pub fn Softmax(comptime T: type, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
    var softmax_items = std.ArrayList(T).initCapacity(alloc, tensor.items.len) catch return TensorError.OutOfMemory;
    var total: T = 0;
    var exp_value: T = undefined;
    for (tensor.items) |item| {
        exp_value = std.math.exp(item);
        softmax_items.append(alloc, exp_value) catch return TensorError.OutOfMemory;
        total += exp_value;
    }
    // normalize all values
    for (0..softmax_items.items.len) |i| {
        softmax_items.items[i] /= total;
    }
    return Tensor(T).init(softmax_items.items, tensor.shape);
}

inline fn setup_test_policy(allocator: Allocator) !FCN(f32) {
    var shape = [_]usize{ 2, 2 };

    const l: Linear(f32) = try .init(allocator, &shape, null);
    return FCN(f32){ .linear = l };
}

test "policy forward happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ 1, 2 };
    var t_shape = [_]usize{2};

    var t = try Tensor(f32).init(&elements, &t_shape);

    var shape = [_]usize{ 2, 2 };

    const l: Linear(f32) = try .init(allocator, &shape, null);
    var fcn = FCN(f32){ .linear = l };
    const mod = fcn.model();
    var p: Policy(f32) = try .init(mod, null, null);

    const output = try p.forward(allocator, &t);
    const exp_shape = [_]usize{2};
    for (exp_shape, output.shape) |e, s| {
        try expect(e == s);
    }
}

test "policy greedy action happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ 1, 2 };
    var t_shape = [_]usize{2};

    var t = try Tensor(f32).init(&elements, &t_shape);
    var fcn = try setup_test_policy(allocator);
    const mod = fcn.model();
    var p: Policy(f32) = try .init(mod, null, null);

    const output = try p.greedy_action(allocator, &t);
    try expect(output < 2 and output >= 0);
}

test "policy sample action happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ 1, 2 };
    var t_shape = [_]usize{2};

    var t = try Tensor(f32).init(&elements, &t_shape);
    var fcn = try setup_test_policy(allocator);
    const mod = fcn.model();
    var p: Policy(f32) = try .init(mod, null, null);

    const output = try p.sample_action(allocator, &t);
    try expect(output < 2 and output >= 0);
}

test "policy epsilon action happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ 1, 2 };
    var t_shape = [_]usize{2};

    var t = try Tensor(f32).init(&elements, &t_shape);
    var fcn = try setup_test_policy(allocator);
    const mod = fcn.model();
    var p: Policy(f32) = try .init(mod, null, null);

    const output = try p.epsilon_action(allocator, &t);
    try expect(output < 2 and output >= 0);
}
