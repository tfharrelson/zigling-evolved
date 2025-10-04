const std = @import("std");

const TensorError = error{
    IncompatibleShapeError,
    HighDimensionality,
};

const IndexTag = enum { all, int };

const Index = union(IndexTag) {
    all: "All",
    int: u32,
};

pub fn Tensor(comptime T: type, items: []T, shape: []u32) !type {
    var expected_num_items = 1;
    for (items) |i| {
        expected_num_items *= i;
    }
    if (items.len != expected_num_items) {
        return TensorError.IncompatibleShapeError;
    }
    return struct {
        const Self = @This();

        items: []T = items,
        shape: []u32 = shape,

        pub fn get(self: *Self, indices: []Index) !Tensor {
            if (indices.len != self.shape) {
                return TensorError.IncompatibleShapeError;
            }
            if (indices.len > 10) {
                return TensorError.HighDimensionality;
            }

            // get the number of elements of the output tensor
            var num_elems: usize = 1;
            var dim: usize = 0;
            for (self.shape, indices) |s, i| {
                switch (i) {
                    IndexTag.all => {
                        num_elems *= s;
                        dim += 1;
                    },
                    IndexTag.int => {
                        num_elems *= 1;
                    },
                }
            }

            var inner_idx = 0;
            var new_shape: [dim]usize = undefined;
            var offset = 0;
            var curr_weight = 1;
            var weights: [dim]usize = undefined;
            for (0..indices.len) |i| {
                const rev_idx = new_shape.len - 1 - i;
                switch (indices[rev_idx]) {
                    IndexTag.all => {
                        new_shape[inner_idx] = self.shape[rev_idx];
                        weights[rev_idx] = curr_weight;
                        inner_idx += 1;
                    },
                    IndexTag.int => {
                        offset += curr_weight * indices[rev_idx];
                    },
                }
                curr_weight *= self.shape[rev_idx];
            }

            // start recursive generation of indices
            // OK, i think i've finally solved something but can't visualize the code, so just gonna state it here
            // for now... one can decompose the the problem into finding an overall offset in the flattened array
            // and iterating over the 'All' columns to find the correct sequence of elements. the problem i'm running
            // into is that i don't know the number of 'All' columns ahead of time - so i don't know how many
            // loops to nest. Still trying to understand if i can get the same effect with modulo math
            // Nvm nvm nvm nvm
            // new plan -> use modulo math, yes i know what i said before, but it's crazy enough to work now
            // the key insight is that we still know how many elements to loop over, no need to dynamically nest
            // for loops. We just loop from 0..num_elems, and follow the modulo math logic. We have to calculate the
            // raw offset on the main array. This is given by the int values in the indices array. Store it for bulk
            // adding to everything later. We can try some simd nonsense b/c we'll have to add it to every index.
            // We also need the weight for each 'All' index. The weight of the right-most 'All' is 1 (e.g. the 1's column)
            // The 2nd from right column is N where N is shape[-1], the 3rd from right column is shape[-2] * shape[-1],
            // etc etc etc. So we will need to loop over the 'All' indices but that is a very small list. To start the
            // modulo math algorithm we need the list of weights, and shape for each 'All' index. Then in the num_elems loop,
            // we take (i % weights[-1]) * all_shape[-1] and add it to a running index. Then we floordivide:
            // (i, all_shape[-1]) and multiply by weights[-2]
            // wait wait wait
            // don't need any weights during the loop, just need to get the indices of the contracted matrix, e.g.
            // if i have a 6x5x4x3x2 tensor and i want a slice over [2,All,0,All,1], then I just need the indices
            // within the 5x3 inner matrix. The algorithm for that is much much simpler. It's a two step using
            // modulo and floordivide. i % N_1 = index_1, floorDivide(i, N_1) => i, loop over the dimensionality
            // of the inner matrix. Then once we get all indices, apply weights to each index value, sum them up
            // add the offset and you have an index in the high dimensional `elements` array on self.

            var elements: [num_elems]T = undefined;
            for (0..num_elems) |i| {
                var elem_idx = 0;
                var floor_idx = i;
                for (0..new_shape.len) |j| {
                    const rev_idx = new_shape.len - 1 - j;
                    elem_idx += @mod(i, new_shape[rev_idx]) * weights[rev_idx];
                    floor_idx = @divFloor(floor_idx, new_shape[rev_idx]);
                }
                elements[i] = self.items[elem_idx + offset];
            }
            return Tensor(T, elements, new_shape);
        }
    };
}
