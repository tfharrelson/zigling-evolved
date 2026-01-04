const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const Model = @import("model.zig").Model;
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("tensor.zig").TensorError;
const Linear = @import("linear.zig").Linear;
const FCN = @import("model.zig").FullyConnectedLayer;

pub fn Graph(comptime T: type) type {
    return struct {
        const Self = @This();

        layers: []Model(T),

        pub fn init(layers: []Model(T)) Self {
            return .{ .layers = layers };
        }

        pub fn forward(self: *Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
            if (self.layers.len == 0) {
                return TensorError.UnexpectedError;
            }
            var x = try self.layers[0].forward(alloc, tensor);
            for (1..self.layers.len) |i| {
                x = try self.layers[i].forward(alloc, &x);
            }
            return x;
        }

        // pub fn backward(self: *Self, alloc: Allocator, tensor: *Tensor) TensorError!Tensor {
        //     // runs the back propagation of gradients back through each individual layer
        //     if (self.layers.len == 0) {
        //         return TensorError.UnexpectedError;
        //     }
        //     var x = try self.layers[-1].forward(alloc, tensor);
        //     for (1..self.layers.len) |i| {
        //         x = try self.layers[self.layers.len - i - 1].backward(alloc, &x);
        //     }
        //     return x;
        // }
    };
}

test "graph forward happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ 1, 2 };
    var t_shape = [_]usize{2};

    var t = try Tensor(f32).init(&elements, &t_shape);
    var shape = [_]usize{ 2, 2 };

    const l: Linear(f32) = try .init(allocator, &shape, null);
    var fcn = FCN(f32){ .linear = l };
    var layers = [_]Model(f32){fcn.model()};
    var p: Graph(f32) = .init(&layers);

    const output = try p.forward(allocator, &t);
    const exp_shape = [_]usize{2};
    for (exp_shape, output.shape) |e, s| {
        try expect(e == s);
    }
}
