const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const meta = std.meta;
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;

/// A MultiGapBuffer stores a list of a struct or tagged union type
/// with a movable gap. Inserting and removal at the gap
/// is O(1), while moving the gap is O(n).
/// Instead of storing a single list of items, MultiGapBuffer
/// stores separate lists for each field of the struct
/// or lists of tags and bare unions.
/// This allows for memory savings if the struct or union has padding,
/// and also improves cache usage if only some fields or just tags
/// are needed for a computation.
pub fn MultiGapBuffer(comptime T: type) type {
    return struct {
        bytes: [*]align(@alignOf(T)) u8 = undefined,
        gap_start: usize = 0,
        gap_end: usize = 0,
        capacity: usize = 0,

        const Elem = switch (@typeInfo(T)) {
            .Struct => T,
            .Union => |u| struct {
                pub const Bare =
                    @Type(.{ .Union = .{
                    .layout = u.layout,
                    .tag_type = null,
                    .fields = u.fields,
                    .decls = &.{},
                } });
                pub const Tag = u.tag_type orelse @compileError("MultiGapBuffer does not support untagged unions");
                tags: Tag,
                data: Bare,

                pub fn fromT(outer: T) @This() {
                    const tag = meta.activeTag(outer);
                    return .{
                        .tags = tag,
                        .data = switch (tag) {
                            inline else => |t| @unionInit(Bare, @tagName(t), @field(outer, @tagName(t))),
                        },
                    };
                }

                pub fn toT(tag: Tag, bare: Bare) T {
                    return switch (tag) {
                        inline else => |t| @unionInit(T, @tagName(t), @field(bare, @tagName(t))),
                    };
                }
            },
            else => @compileError("MultiGapBuffer only supports structs and tagged unions"),
        };

        pub const Field = meta.FieldEnum(Elem);

        /// A MultiGapBuffer.Slice contains cached start pointers for each field in the buffer.
        /// These pionters are not normally stored to reduce the size of the buffer in memory.
        /// If you are accessing multiple fields, call `slice()` first to compute the pointers,
        /// and then get the field arrays from the slice.
        pub const Slice = struct {
            /// This array is indexed by the field index
            /// which can be obtained by using @intFromEnum() on the Field enum
            ptrs: [fields.len][*]u8,
            gap_start: usize,
            gap_end: usize,
            capacity: usize,

            pub fn firstHalf(self: Slice, comptime field: Field) []FieldType(field) {
                if (self.capacity == 0) {
                    return &.{};
                }
                const casted_ptr = self.getPtr(field);
                return casted_ptr[0..self.gap_start];
            }

            pub fn secondHalf(self: Slice, comptime field: Field) []FieldType(field) {
                if (self.capacity == 0) {
                    return &.{};
                }
                const casted_ptr = self.getPtr(field);
                return casted_ptr[self.gap_end..self.capacity];
            }

            pub fn realIndex(self: Slice, index: usize) usize {
                return if (index < self.gap_start) index else self.gap_end + (index - self.gap_start);
            }

            pub fn realLength(self: Slice) usize {
                return self.gap_start + (self.capacity - self.gap_end);
            }

            /// caller owns the allocated memory.
            pub fn dupeLogicalSlice(self: Slice, allocator: Allocator, comptime field: Field, logical_start: usize, len: usize) Allocator.Error![]FieldType(field) {
                const new_mem = try allocator.alloc(FieldType(field), len);
                const casted_ptr = self.getPtr(field);
                const start = self.realIndex(logical_start);
                if (logical_start < self.gap_start and logical_start + len >= self.gap_start) {
                    // our desired slice jumps the gap
                    const first_len = self.gap_start - logical_start;
                    @memcpy(new_mem[0..first_len], casted_ptr[start..self.gap_start]);
                    @memcpy(new_mem[first_len..], casted_ptr[self.gap_end..][0 .. len - first_len]);
                } else {
                    // our desired logical slice does not jump the gap
                    @memcpy(new_mem, casted_ptr[start..][0..len]);
                }
                return new_mem;
            }

            pub fn set(self: *Slice, index: usize, elem: T) void {
                const e = switch (@typeInfo(T)) {
                    .Struct => elem,
                    .Union => Elem.fromT(elem),
                    else => unreachable,
                };
                const j = self.realIndex(index);
                inline for (fields, 0..) |field_info, i| {
                    const field: Field = @enumFromInt(i);
                    const casted_ptr = self.getPtr(field);
                    casted_ptr[j] = @field(e, field_info.name);
                }
            }

            pub fn get(self: Slice, index: usize) T {
                var result: Elem = undefined;
                const j = self.realIndex(index);
                inline for (fields, 0..) |field_info, i| {
                    const field: Field = @enumFromInt(i);
                    const casted_ptr = self.getPtr(field);
                    @field(result, field_info.name) = casted_ptr[j];
                }
                return switch (@typeInfo(T)) {
                    .Struct => result,
                    .Union => Elem.toT(result.tags, result.data),
                    else => unreachable,
                };
            }

            pub fn moveGap(self: *Slice, new_start: usize) void {
                if (new_start == self.gap_start) return;
                const len = self.realLength();
                assert(new_start <= len);
                if (new_start < self.gap_start) {
                    const len_moved = self.gap_start - new_start;
                    // we're moving items _backwards_
                    inline for (0..fields.len) |i| {
                        const field: Field = @enumFromInt(i);
                        const casted_ptr = self.getPtr(field);
                        std.mem.copyBackwards(
                            FieldType(field),
                            casted_ptr[self.gap_end - len_moved .. self.gap_end],
                            casted_ptr[new_start..self.gap_start],
                        );
                    }
                    self.gap_start = new_start;
                    self.gap_end -= len_moved;
                } else {
                    const len_moved = new_start - self.gap_start;
                    // we're moving items _forwards_
                    inline for (0..fields.len) |i| {
                        const field: Field = @enumFromInt(i);
                        const casted_ptr = self.getPtr(field);
                        std.mem.copyForwards(
                            FieldType(field),
                            casted_ptr[self.gap_start..new_start],
                            casted_ptr[self.gap_end .. self.gap_end + len_moved],
                        );
                    }
                    self.gap_start = new_start;
                    self.gap_end += len_moved;
                }
            }

            pub fn toMultiGapBuffer(self: Slice) Self {
                if (self.capacity == 0) {
                    return .{};
                }
                const unaligned_ptr = self.ptrs[sizes.fields[0]];
                const aligned_ptr: [*]align(@alignOf(Elem)) u8 = @alignCast(unaligned_ptr);
                return .{
                    .bytes = aligned_ptr,
                    .gap_start = self.gap_start,
                    .gap_end = self.gap_end,
                    .capacity = self.capacity,
                };
            }

            pub fn deinit(self: *Slice, allocator: Allocator) void {
                var other = self.toMultiGapBuffer();
                other.deinit(allocator);
                self.* = undefined;
            }

            pub fn getField(self: Slice, comptime field: Field, idx: usize) FieldType(field) {
                const F = FieldType(field);
                const byte_ptr = self.ptrs[@intFromEnum(field)];
                const casted_ptr: [*]F = if (@sizeOf(F) == 0)
                    undefined
                else
                    @ptrCast(@alignCast(byte_ptr));
                const real_idx = self.realIndex(idx);
                return casted_ptr[real_idx];
            }

            pub fn getPtr(self: Slice, comptime field: Field) [*]FieldType(field) {
                const F = FieldType(field);
                const byte_ptr = self.ptrs[@intFromEnum(field)];
                const casted_ptr: [*]F = if (@sizeOf(F) == 0)
                    undefined
                else
                    @ptrCast(@alignCast(byte_ptr));
                return casted_ptr;
            }

            /// This function is used in the debugger pretty formatters to fetch the
            /// child field order and entry type to facilitate fancy debug printing for this type.
            fn dbHelper(self: *Slice, child: *Elem, field: *Field, entry: *Entry) void {
                _ = self; // autofix
                _ = child; // autofix
                _ = field; // autofix
                _ = entry; // autofix
            }
        };

        const Self = @This();

        const fields = meta.fields(Elem);
        /// `sizes.bytes` is an array of @sizeOf each T field. Sorted by alignment, descending.
        /// `sizes.fields` is an array mapping from `sizes.bytes` array index to field index
        const sizes = blk: {
            const Data = struct {
                size: usize,
                size_index: usize,
                alignment: usize,
            };
            var data: [fields.len]Data = undefined;
            for (fields, 0..) |field_info, i| {
                data[i] = .{
                    .size = @sizeOf(field_info.type),
                    .size_index = i,
                    .alignment = if (@sizeOf(field_info.type) == 0) 1 else field_info.alignment,
                };
            }
            const Sort = struct {
                fn lessThan(_: void, lhs: Data, rhs: Data) bool {
                    return lhs.alignment > rhs.alignment;
                }
            };
            mem.sort(Data, &data, {}, Sort.lessThan);
            var size_bytes: [fields.len]usize = undefined;
            var field_indices: [fields.len]usize = undefined;
            for (data, 0..) |elem, i| {
                size_bytes[i] = elem.size;
                field_indices[i] = elem.size_index;
            }
            break :blk .{
                .bytes = size_bytes,
                .fields = field_indices,
            };
        };

        /// Release all allocated memory.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            allocator.free(self.allocatedBytes());
            self.* = undefined;
        }

        /// The caller owns the returned memory. Empties this MultiArrayList.
        pub fn toOwnedSlice(self: *Self) Slice {
            const result = self.slice();
            self.* = .{};
            return result;
        }

        /// Compute pointers to the start of each field of the array.
        /// If you need to access multiple fields, calling this may
        /// be more efficient than calling `firstHalf()` or `secondHalf()` multiple times.
        pub fn slice(self: Self) Slice {
            var result: Slice = .{
                .ptrs = undefined,
                .gap_start = self.gap_start,
                .gap_end = self.gap_end,
                .capacity = self.capacity,
            };
            var ptr: [*]u8 = self.bytes;
            for (sizes.bytes, sizes.fields) |field_size, i| {
                result.ptrs[i] = ptr;
                ptr += field_size * self.capacity;
            }
            return result;
        }

        /// Get the slice of values for a specified field.
        /// If you need multiple fields, consider calling slice()
        /// instead.
        pub fn firstHalf(self: Self, comptime field: Field) []FieldType(field) {
            return self.slice().firstHalf(field);
        }

        /// Get the slice of values for a specified field.
        /// If you need multiple fields, consider calling slice()
        /// instead.
        pub fn secondHalf(self: Self, comptime field: Field) []FieldType(field) {
            return self.slice().secondHalf(field);
        }

        /// Overwrite one array element with new data.
        pub fn set(self: *Self, index: usize, elem: T) void {
            var slices = self.slice();
            slices.set(index, elem);
        }

        /// Obtain all the data for one array element.
        pub fn get(self: Self, index: usize) T {
            return self.slice().get(index);
        }

        /// gets the logical length of the buffer
        pub fn realLength(self: Self) usize {
            return self.gap_start + (self.capacity - self.gap_end);
        }

        /// gets an offset from a logical index
        pub fn realIndex(self: Self, index: usize) usize {
            return if (index < self.gap_start) index else self.gap_end + (index - self.gap_start);
        }

        /// moves the gap to the given logical index
        pub fn moveGap(self: *Self, new_start: usize) void {
            var slices = self.slice();
            slices.moveGap(new_start);
            self.gap_end = slices.gap_end;
            self.gap_start = slices.gap_start;
        }

        /// Extend the buffer by 1 element after the gap. Allocates more memory as necessary.
        pub fn appendAfter(self: *Self, allocator: Allocator, elem: T) !void {
            try self.ensureUnusedCapacity(allocator, 1);
            self.appendAfterAssumeCapacity(elem);
        }

        /// Extend the buffer by 1 element after the gap, but asserting `self.capacity`
        /// is sufficient to hold an additional item.
        pub fn appendAfterAssumeCapacity(self: *Self, elem: T) void {
            assert(self.realLength() < self.capacity);
            self.gap_end -= 1;
            // gap_end is the same logical index as gap start,
            // and set needs a logical index
            self.set(self.gap_start, elem);
        }

        /// Extend the buffer by 1 element after the gap, returning the newly reserved
        /// index with uninitialized data.
        /// Allocates more memory as necesasry.
        pub fn addOneAfter(self: *Self, allocator: Allocator) Allocator.Error!usize {
            try self.ensureUnusedCapacity(allocator, 1);
            return self.addOneAfterAssumeCapacity();
        }

        /// Extend the buffer by 1 element after the gap, asserting `self.capacity`
        /// is sufficient to hold an additional item.  Returns the
        /// newly reserved index with uninitialized data.
        pub fn addOneAfterAssumeCapacity(self: *Self) usize {
            assert(self.realLength() < self.capacity);
            self.gap_end -= 1;
            return self.gap_end;
        }

        /// Extend the buffer by 1 element before the gap. Allocates more memory as necessary.
        pub fn appendBefore(self: *Self, allocator: Allocator, elem: T) !void {
            try self.ensureUnusedCapacity(allocator, 1);
            self.appendBeforeAssumeCapacity(elem);
        }

        /// Extend the buffer by 1 element before the gap, but asserting `self.capacity`
        /// is sufficient to hold an additional item.
        pub fn appendBeforeAssumeCapacity(self: *Self, elem: T) void {
            assert(self.realLength() < self.capacity);
            self.gap_start += 1;
            self.set(self.gap_start - 1, elem);
        }

        /// Extend the buffer by 1 element before the gap, returning the newly reserved
        /// index with uninitialized data.
        /// Allocates more memory as necesasry.
        pub fn addOneBefore(self: *Self, allocator: Allocator) Allocator.Error!usize {
            try self.ensureUnusedCapacity(allocator, 1);
            return self.addOneBeforeAssumeCapacity();
        }

        /// Extend the buffer by 1 element before the gap, asserting `self.capacity`
        /// is sufficient to hold an additional item.  Returns the
        /// newly reserved index with uninitialized data.
        pub fn addOneBeforeAssumeCapacity(self: *Self) usize {
            assert(self.realLength() < self.capacity);
            const index = self.gap_start;
            self.gap_start += 1;
            return index;
        }

        /// Remove and return the last element after the gap from the buffer.
        /// Asserts the buffer has at least one item.
        /// Invalidates pointers to fields of the removed element.
        pub fn popAfter(self: *Self) T {
            const val = self.get(self.gap_start);
            self.gap_end += 1;
            return val;
        }

        /// Remove and return the last element after the gap from the buffer, or
        /// return `null` if there is none.
        /// Invalidates pointers to fields of the removed element, if any.
        pub fn popAfterOrNull(self: *Self) ?T {
            if (self.gap_end == self.capacity) return null;
            return self.popAfter();
        }

        /// Remove and return the last element before the gap from the buffer.
        /// Asserts the buffer has at least one item.
        /// Invalidates pointers to fields of the removed element.
        pub fn popBefore(self: *Self) T {
            const val = self.get(self.gap_start - 1);
            self.gap_start -= 1;
            return val;
        }

        /// Remove and return the last element before the gap from the buffer, or
        /// return `null` if there is none.
        /// Invalidates pointers to fields of the removed element, if any.
        pub fn popBeforeOrNull(self: *Self) ?T {
            if (self.gap_start == 0) return null;
            return self.popBefore();
        }

        /// Inserts an item into an ordered buffer.  Shifts all elements
        /// after and including the specified index back by one and
        /// sets the given index to the specified element.  May reallocate
        /// and invalidate iterators.
        /// O(1) if the gap does not move.
        pub fn insert(self: *Self, allocator: Allocator, index: usize, elem: T) !void {
            try self.ensureUnusedCapacity(allocator, 1);
            self.insertAssumeCapacity(index, elem);
        }

        /// Inserts an item into an ordered buffer which has room for it.
        /// Shifts all elements after and including the specified index
        /// back by one and sets the given index to the specified element.
        /// Will not reallocate the array, does not invalidate iterators.
        /// O(1) if the gap does not move
        pub fn insertAssumeCapacity(self: *Self, index: usize, elem: T) void {
            assert(self.realLength() < self.capacity);
            assert(index <= self.realLength());
            self.moveGap(index);
            self.appendBeforeAssumeCapacity(elem);
        }

        /// Remove the specified item from the buffer, shifting items
        /// after it to preserve order.
        /// Moves the gap to `index`.
        /// This operation is O(1) if the gap does not move.
        pub fn orderedRemoveAfter(self: *Self, index: usize) void {
            self.moveGap(index);
            _ = self.popAfter();
        }

        /// Remove the specified item from the buffer, shifting items
        /// after it to preserve order.
        /// Moves the gap to `index + 1`.
        /// This operation is O(1) if the gap does not move.
        pub fn orderedRemoveBefore(self: *Self, index: usize) void {
            self.moveGap(index + 1);
            _ = self.popBefore();
        }

        /// Remove the specified item from the buffer, swapping the last
        /// item in the buffer after the gap into its position.  Fast, but does not
        /// retain buffer ordering.
        /// asserts that there is an element after the gap
        pub fn swapRemoveAfter(self: *Self, index: usize) void {
            const j = self.realIndex(index);
            const slices = self.slice();
            inline for (fields, 0..) |_, i| {
                const field: Field = @enumFromInt(i);
                const casted_ptr = slices.getPtr(field);
                casted_ptr[j] = casted_ptr[self.gap_end];
                casted_ptr[self.gap_end] = undefined;
            }
            self.gap_end += 1;
        }

        /// Remove the specified item from the buffer, swapping the last
        /// item in the buffer before the gap into its position.  Fast, but does not
        /// retain buffer ordering.
        /// asserts that there is an element before the gap
        pub fn swapRemoveBefore(self: *Self, index: usize) void {
            const j = self.realIndex(index);
            const slices = self.slice();
            inline for (fields, 0..) |_, i| {
                const field: Field = @enumFromInt(i);
                const casted_ptr = slices.getPtr(field);
                casted_ptr[j] = casted_ptr[self.gap_start - 1];
                casted_ptr[self.gap_start - 1] = undefined;
            }
            self.gap_start -= 1;
        }

        /// Adjust the buffer's realLength to `new_len`,
        /// by adding elements after the gap.
        /// Does not initialize added items, if any.
        pub fn resizeAfter(self: *Self, allocator: Allocator, new_len: usize) !void {
            const n = self.realLength() -| new_len;
            try self.ensureTotalCapacity(allocator, new_len);
            self.gap_end -= n;
        }

        /// Adjust the buffer's realLength to `new_len`,
        /// by adding elements before the gap.
        /// Does not initialize added items, if any.
        pub fn resizeBefore(self: *Self, allocator: Allocator, new_len: usize) !void {
            const n = self.realLength() -| new_len;
            try self.ensureTotalCapacity(allocator, new_len);
            self.gap_start += n;
        }

        /// Attempt to reduce allocated capacity to `new_len`.
        /// If `new_len` is greater than zero, this may fail to reduce the capacity,
        /// but the data remains intact and the length is updated to new_len by discarding elements after the gap.
        pub fn shrinkAndFreeAfter(self: *Self, allocator: Allocator, new_len: usize) void {
            if (new_len == 0) {
                allocator.free(self.allocatedBytes());
                self.* = .{};
                return;
            }
            assert(new_len <= self.capacity);
            assert(new_len <= self.realLength());
            const lost = self.realLength() - new_len;
            assert(self.gap_end + lost <= self.capacity);
            const new_gap_end = self.gap_end + lost;

            const other_bytes = allocator.alignedAlloc(
                u8,
                @alignOf(Elem),
                capacityInBytes(new_len),
            ) catch {
                const self_slice = self.slice();
                inline for (fields, 0..) |field_info, i| {
                    if (@sizeOf(field_info.type) != 0) {
                        const field: Field = @enumFromInt(i);
                        const dest_slice = self_slice.firstHalf(field)[0..lost];
                        // We use memset here for more efficient codegen in safety-checked,
                        // valgrind-enabled builds. Otherwise the valgrind client request
                        // will be repeated for every element.
                        @memset(dest_slice, undefined);
                    }
                }
                self.gap_end = new_gap_end;
                return;
            };
            var other = Self{
                .bytes = other_bytes.ptr,
                .capacity = new_len,
                .gap_start = self.gap_start,
                .gap_end = self.gap_start,
            };
            self.gap_end = new_gap_end;
            const self_slice = self.slice();
            const other_slice = other.slice();
            inline for (fields, 0..) |field_info, i| {
                if (@sizeOf(field_info.type) != 0) {
                    const field = @as(Field, @enumFromInt(i));
                    @memcpy(other_slice.firstHalf(field), self_slice.firstHalf(field));
                    @memcpy(other_slice.secondHalf(field), self_slice.secondHalf(field));
                }
            }
            allocator.free(self.allocatedBytes());
            self.* = other;
        }

        /// Attempt to reduce allocated capacity to `new_len`.
        /// If `new_len` is greater than zero, this may fail to reduce the capacity,
        /// but the data remains intact and the length is updated to new_len by discarding elements before the gap.
        pub fn shrinkAndFreeBefore(self: *Self, allocator: Allocator, new_len: usize) void {
            if (new_len == 0) {
                allocator.free(self.allocatedBytes());
                self.* = .{};
                return;
            }
            assert(new_len <= self.capacity);
            assert(new_len <= self.realLength());
            const lost = self.realLength() - new_len;
            assert(self.gap_start >= lost);
            const new_gap_start = self.gap_start - lost;

            const other_bytes = allocator.alignedAlloc(
                u8,
                @alignOf(Elem),
                capacityInBytes(new_len),
            ) catch {
                const self_slice = self.slice();
                inline for (fields, 0..) |field_info, i| {
                    if (@sizeOf(field_info.type) != 0) {
                        const field: Field = @enumFromInt(i);
                        const dest_slice = self_slice.firstHalf(field)[new_gap_start..];
                        // We use memset here for more efficient codegen in safety-checked,
                        // valgrind-enabled builds. Otherwise the valgrind client request
                        // will be repeated for every element.
                        @memset(dest_slice, undefined);
                    }
                }
                self.gap_start = new_gap_start;
                return;
            };
            var other = Self{
                .bytes = other_bytes.ptr,
                .capacity = new_len,
                .gap_start = new_gap_start,
                .gap_end = new_gap_start,
            };
            self.gap_start = new_gap_start;
            const self_slice = self.slice();
            const other_slice = other.slice();
            inline for (fields, 0..) |field_info, i| {
                if (@sizeOf(field_info.type) != 0) {
                    const field = @as(Field, @enumFromInt(i));
                    @memcpy(other_slice.firstHalf(field), self_slice.firstHalf(field));
                    @memcpy(other_slice.secondHalf(field), self_slice.secondHalf(field));
                }
            }
            allocator.free(self.allocatedBytes());
            self.* = other;
        }

        /// Reduce realLength() to `new_len` by discarding after the gap.
        /// Asserts that this is possible
        /// Invalidates pointers to elements `items[new_len..]`.
        /// Keeps capacity the same.
        pub fn shrinkAfterRetainingCapacity(self: *Self, new_len: usize) void {
            assert(self.realLength() >= new_len);
            const lost = self.realLength() - new_len;
            assert(self.gap_end + lost <= self.capacity);
            self.gap_end += lost;
        }

        /// Reduce realLength() to `new_len` by discarding from before the gap.
        /// Asserts that this is possible
        /// Invalidates pointers to elements `items[new_len..]`.
        /// Keeps capacity the same.
        pub fn shrinkBeforeRetainingCapacity(self: *Self, new_len: usize) void {
            assert(self.realLength() >= new_len);
            const lost = self.realLength() - new_len;
            assert(self.gap_start >= lost);
            self.gap_start -= lost;
        }

        /// Modify the buffer so that it can hold at least `new_capacity` items.
        /// Implements super-linear growth to achieve amortized O(1) append operations.
        /// Invalidates pointers if additional memory is needed.
        pub fn ensureTotalCapacity(self: *Self, allocator: Allocator, new_capacity: usize) !void {
            var better_capacity = self.capacity;
            if (better_capacity >= new_capacity) return;

            while (true) {
                better_capacity += better_capacity / 2 + 8;
                if (better_capacity >= new_capacity) break;
            }

            return self.setCapacity(allocator, better_capacity);
        }

        /// Modify the buffer so that it can hold at least `additional_count` **more** items.
        /// Invalidates pointers if additional memory is needed.
        pub fn ensureUnusedCapacity(self: *Self, allocator: Allocator, additional_count: usize) !void {
            return self.ensureTotalCapacity(allocator, self.realLength() + additional_count);
        }

        /// Modify the buffer so that it can hold exactly `new_capacity` items.
        /// Invalidates pointers if additional memory is needed.
        /// `new_capacity` must be greater or equal to `len`.
        pub fn setCapacity(self: *Self, allocator: Allocator, new_capacity: usize) !void {
            assert(new_capacity >= self.realLength());
            const new_bytes = try allocator.alignedAlloc(
                u8,
                @alignOf(Elem),
                capacityInBytes(new_capacity),
            );
            if (self.realLength() == 0) {
                allocator.free(self.allocatedBytes());
                self.bytes = new_bytes.ptr;
                self.capacity = new_capacity;
                self.gap_end = new_capacity;
                return;
            }
            const second_half_len = self.realLength() - self.gap_start;
            var other = Self{
                .bytes = new_bytes.ptr,
                .capacity = new_capacity,
                .gap_end = new_capacity - second_half_len,
                .gap_start = self.gap_start,
            };
            const self_slice = self.slice();
            const other_slice = other.slice();
            inline for (fields, 0..) |field_info, i| {
                if (@sizeOf(field_info.type) != 0) {
                    const field: Field = @enumFromInt(i);
                    @memcpy(other_slice.firstHalf(field), self_slice.firstHalf(field));
                    @memcpy(other_slice.secondHalf(field), self_slice.secondHalf(field));
                }
            }
            allocator.free(self.allocatedBytes());
            self.* = other;
        }

        /// Create a copy of this buffer with a new backing store,
        /// using the specified allocator.
        pub fn clone(self: Self, allocator: Allocator) !Self {
            var result = Self{};
            errdefer result.deinit(allocator);
            try result.ensureTotalCapacity(allocator, self.len);
            result.gap_end = self.gap_end;
            result.gap_start = self.gap_start;
            const self_slice = self.slice();
            const result_slice = result.slice();
            inline for (fields, 0..) |field_info, i| {
                if (@sizeOf(field_info.type) != 0) {
                    const field: Field = @enumFromInt(i);
                    @memcpy(result_slice.firstHalf(field), self_slice.firstHalf(field));
                    @memcpy(result_slice.secondHalf(field), self_slice.secondHalf(field));
                }
            }
            return result;
        }

        /// `ctx` has the following method:
        /// `fn lessThan(ctx: @TypeOf(ctx), a_index: usize, b_index: usize) bool`
        fn sortInternal(self: Self, a: usize, b: usize, ctx: anytype, comptime mode: std.sort.Mode) void {
            const sort_context: struct {
                sub_ctx: @TypeOf(ctx),
                slice: Slice,

                pub fn swap(sc: @This(), a_index: usize, b_index: usize) void {
                    const ra = sc.slice.realIndex(a_index);
                    const rb = sc.slice.realIndex(b_index);
                    inline for (fields, 0..) |field_info, i| {
                        if (@sizeOf(field_info.type) != 0) {
                            const field: Field = @enumFromInt(i);
                            const ptr = sc.slice.getPtr(field);
                            mem.swap(field_info.type, &ptr[ra], &ptr[rb]);
                        }
                    }
                }

                pub fn lessThan(sc: @This(), a_index: usize, b_index: usize) bool {
                    return sc.sub_ctx.lessThan(a_index, b_index);
                }
            } = .{
                .sub_ctx = ctx,
                .slice = self.slice(),
            };

            switch (mode) {
                .stable => mem.sortContext(a, b, sort_context),
                .unstable => mem.sortUnstableContext(a, b, sort_context),
            }
        }

        /// This function guarantees a stable sort, i.e the relative order of equal elements is preserved during sorting.
        /// Read more about stable sorting here: https://en.wikipedia.org/wiki/Sorting_algorithm#Stability
        /// If this guarantee does not matter, `sortUnstable` might be a faster alternative.
        /// `ctx` has the following method:
        /// `fn lessThan(ctx: @TypeOf(ctx), a_index: usize, b_index: usize) bool`
        pub fn sort(self: Self, ctx: anytype) void {
            self.sortInternal(0, self.len, ctx, .stable);
        }

        /// Sorts only the subsection of items between indices `a` and `b` (excluding `b`)
        /// This function guarantees a stable sort, i.e the relative order of equal elements is preserved during sorting.
        /// Read more about stable sorting here: https://en.wikipedia.org/wiki/Sorting_algorithm#Stability
        /// If this guarantee does not matter, `sortSpanUnstable` might be a faster alternative.
        /// `ctx` has the following method:
        /// `fn lessThan(ctx: @TypeOf(ctx), a_index: usize, b_index: usize) bool`
        pub fn sortSpan(self: Self, a: usize, b: usize, ctx: anytype) void {
            self.sortInternal(a, b, ctx, .stable);
        }

        /// This function does NOT guarantee a stable sort, i.e the relative order of equal elements may change during sorting.
        /// Due to the weaker guarantees of this function, this may be faster than the stable `sort` method.
        /// Read more about stable sorting here: https://en.wikipedia.org/wiki/Sorting_algorithm#Stability
        /// `ctx` has the following method:
        /// `fn lessThan(ctx: @TypeOf(ctx), a_index: usize, b_index: usize) bool`
        pub fn sortUnstable(self: Self, ctx: anytype) void {
            self.sortInternal(0, self.len, ctx, .unstable);
        }
        /// Sorts only the subsection of items between indices `a` and `b` (excluding `b`)
        /// This function does NOT guarantee a stable sort, i.e the relative order of equal elements may change during sorting.
        /// Due to the weaker guarantees of this function, this may be faster than the stable `sortSpan` method.
        /// Read more about stable sorting here: https://en.wikipedia.org/wiki/Sorting_algorithm#Stability
        /// `ctx` has the following method:
        /// `fn lessThan(ctx: @TypeOf(ctx), a_index: usize, b_index: usize) bool`
        pub fn sortSpanUnstable(self: Self, a: usize, b: usize, ctx: anytype) void {
            self.sortInternal(a, b, ctx, .unstable);
        }

        fn capacityInBytes(capacity: usize) usize {
            comptime var elem_bytes: usize = 0;
            inline for (sizes.bytes) |size| elem_bytes += size;
            return elem_bytes * capacity;
        }

        fn allocatedBytes(self: Self) []align(@alignOf(Elem)) u8 {
            return self.bytes[0..capacityInBytes(self.capacity)];
        }

        fn FieldType(comptime field: Field) type {
            return meta.fieldInfo(Elem, field).type;
        }

        const Entry = entry: {
            var entry_fields: [fields.len]std.builtin.Type.StructField = undefined;
            for (&entry_fields, sizes.fields) |*entry_field, i|
                entry_field.* = .{
                    .name = fields[i].name ++ "_ptr",
                    .type = *fields[i].type,
                    .default_value = null,
                    .is_comptime = fields[i].is_comptime,
                    .alignment = fields[i].alignment,
                };
            break :entry @Type(.{ .Struct = .{
                .layout = .@"extern",
                .fields = &entry_fields,
                .decls = &.{},
                .is_tuple = false,
            } });
        };

        /// This function is used in the debugger pretty formatters to fetch the
        /// child field order and entry type to facilitate fancy debug printing for this type.
        fn dbHelper(self: *Self, child: *Elem, field: *Field, entry: *Entry) void {
            _ = self; // autofix
            _ = child; // autofix
            _ = field; // autofix
            _ = entry; // autofix
        }

        comptime {
            if (!builtin.strip_debug_info) {
                _ = &dbHelper;
                _ = &Slice.dbHelper;
            }
        }
    };
}

test "basic usage" {
    const a = testing.allocator;

    const Foo = struct {
        a: u32,
        b: []const u8,
        c: u8,
    };

    var buffer: MultiGapBuffer(Foo) = .{};
    defer buffer.deinit(a);

    try testing.expectEqual(0, buffer.realLength());
    try buffer.ensureTotalCapacity(a, 2);

    buffer.appendBeforeAssumeCapacity(.{
        .a = 1,
        .b = "foobar",
        .c = 'a',
    });
    buffer.appendAfterAssumeCapacity(.{
        .a = 2,
        .b = "zigzag",
        .c = 'b',
    });
    try testing.expectEqual(2, buffer.realLength());
    try testing.expectEqualStrings("foobar", buffer.firstHalf(.b)[0]);
    try testing.expectEqualStrings("zigzag", buffer.secondHalf(.b)[0]);

    try buffer.appendBefore(a, .{
        .a = 3,
        .b = "fizzbuzz",
        .c = 'c',
    });
    try testing.expectEqualSlices(u32, buffer.firstHalf(.a), &.{ 1, 3 });
    try testing.expectEqualSlices(u8, buffer.firstHalf(.c), &.{ 'a', 'c' });
    try testing.expectEqual(3, buffer.realLength());
    try testing.expectEqual(2, buffer.gap_start);
    try testing.expectEqualStrings("foobar", buffer.firstHalf(.b)[0]);
    try testing.expectEqualStrings("fizzbuzz", buffer.firstHalf(.b)[1]);
    try testing.expectEqualStrings("zigzag", buffer.secondHalf(.b)[0]);

    for (0..6) |i| {
        try buffer.appendAfter(a, .{
            .a = @intCast(4 + i),
            .b = "whatever",
            .c = @intCast('d' + i),
        });
    }
    try testing.expectEqualSlices(
        u32,
        &.{ 9, 8, 7, 6, 5, 4, 2 },
        buffer.secondHalf(.a),
    );
    try testing.expectEqualSlices(
        u8,
        &.{ 'i', 'h', 'g', 'f', 'e', 'd', 'b' },
        buffer.secondHalf(.c),
    );
    buffer.shrinkAndFreeAfter(a, 3);
    buffer.moveGap(buffer.realLength());
    try testing.expectEqual(3, buffer.realLength());
    try testing.expectEqualSlices(u32, &.{ 1, 3, 2 }, buffer.firstHalf(.a));
    try testing.expectEqualSlices(u8, &.{ 'a', 'c', 'b' }, buffer.firstHalf(.c));
    try testing.expectEqualStrings("foobar", buffer.firstHalf(.b)[0]);
    try testing.expectEqualStrings("fizzbuzz", buffer.firstHalf(.b)[1]);
    try testing.expectEqualStrings("zigzag", buffer.firstHalf(.b)[2]);

    buffer.set(try buffer.addOneBefore(a), .{
        .a = 4,
        .b = "xnopyt",
        .c = 'd',
    });
    try testing.expectEqualStrings("xnopyt", buffer.popBefore().b);
    try testing.expectEqual(null, buffer.popAftereOrNull());
    try testing.expectEqual('b', if (buffer.popBeforeOrNull()) |e| e.c else null);
    try testing.expectEqual(3, buffer.popBefore().a);
    try testing.expectEqual('a', buffer.popBefore().c);
    try testing.expectEqual(null, buffer.popBeforeOrNull());
}

test "ensure capacity on empty list" {
    const a = testing.allocator;

    const Foo = struct {
        a: u32,
        b: u8,
    };
    var buffer = MultiGapBuffer(Foo){};
    defer buffer.deinit(a);

    try buffer.ensureTotalCapacity(a, 2);
    buffer.appendBeforeAssumeCapacity(.{ .a = 1, .b = 2 });
    buffer.appendBeforeAssumeCapacity(.{ .a = 3, .b = 4 });
    try testing.expectEqualSlices(u32, &.{ 1, 3 }, buffer.firstHalf(.a));
    try testing.expectEqualSlices(u8, &.{ 2, 4 }, buffer.firstHalf(.b));

    buffer.gap_start = 0;
    buffer.appendAfterAssumeCapacity(.{ .a = 7, .b = 8 });
    buffer.appendAfterAssumeCapacity(.{ .a = 5, .b = 6 });

    try testing.expectEqualSlices(u32, &.{ 5, 7 }, buffer.secondHalf(.a));
    try testing.expectEqualSlices(u8, &.{ 6, 8 }, buffer.secondHalf(.b));

    buffer.gap_end = buffer.capacity;
    buffer.appendBeforeAssumeCapacity(.{ .a = 9, .b = 10 });
    buffer.appendBeforeAssumeCapacity(.{ .a = 11, .b = 12 });
    try testing.expectEqualSlices(u32, &.{ 9, 11 }, buffer.firstHalf(.a));
    try testing.expectEqualSlices(u8, &.{ 10, 12 }, buffer.firstHalf(.b));
}

test "insert elements" {
    const a = testing.allocator;

    const Foo = struct {
        a: u8,
        b: u32,
    };

    var buffer: MultiGapBuffer(Foo) = .{};
    defer buffer.deinit(a);

    try buffer.insert(a, 0, .{ .a = 1, .b = 2 });
    try buffer.ensureUnusedCapacity(a, 1);
    buffer.insertAssumeCapacity(1, .{ .a = 2, .b = 3 });

    buffer.moveGap(buffer.realLength());
    try testing.expectEqualSlices(u8, &.{ 1, 2 }, buffer.firstHalf(.a));
    try testing.expectEqualSlices(u32, &.{ 2, 3 }, buffer.firstHalf(.b));
}

test "union" {
    const a = testing.allocator;

    const Foo = union(enum) {
        a: u32,
        b: []const u8,
    };

    var buffer: MultiGapBuffer(Foo) = .{};
    defer buffer.deinit(a);

    try testing.expectEqual(0, buffer.realLength());

    try buffer.ensureTotalCapacity(a, 2);

    buffer.appendBeforeAssumeCapacity(.{ .a = 1 });
    buffer.appendBeforeAssumeCapacity(.{ .b = "zigzag" });

    try testing.expectEqualSlices(meta.Tag(Foo), &.{ .a, .b }, buffer.firstHalf(.tags));
    try testing.expectEqual(2, buffer.firstHalf(.tags).len);

    buffer.appendAfterAssumeCapacity(.{ .b = "foobar" });
    try testing.expectEqualStrings("zigzag", buffer.firstHalf(.data)[1].b);
    try testing.expectEqualStrings("foobar", buffer.secondHalf(.data)[0].b);

    // Add 6 more things to force a capacity increase.
    for (0..6) |i| {
        try buffer.appendBefore(a, .{ .a = @intCast(4 + i) });
    }

    try testing.expectEqualSlices(
        meta.Tag(Foo),
        &.{ .a, .b, .a, .a, .a, .a, .a, .a },
        buffer.firstHalf(.tags),
    );
    try testing.expectEqual(Foo{ .a = 1 }, buffer.get(0));
    try testing.expectEqual(Foo{ .b = "zigzag" }, buffer.get(1));
    try testing.expectEqual(Foo{ .b = "foobar" }, buffer.get(8));
    try testing.expectEqual(Foo{ .a = 4 }, buffer.get(2));
    try testing.expectEqual(Foo{ .a = 5 }, buffer.get(3));
    try testing.expectEqual(Foo{ .a = 6 }, buffer.get(4));
    try testing.expectEqual(Foo{ .a = 7 }, buffer.get(5));
    try testing.expectEqual(Foo{ .a = 8 }, buffer.get(6));
    try testing.expectEqual(Foo{ .a = 9 }, buffer.get(7));

    buffer.shrinkAndFreeBefore(a, 3);
    buffer.moveGap(buffer.realLength());

    try testing.expectEqual(3, buffer.firstHalf(.tags).len);
    try testing.expectEqualSlices(meta.Tag(Foo), &.{ .a, .b, .b }, buffer.firstHalf(.tags));

    try testing.expectEqual(Foo{ .a = 1 }, buffer.get(0));
    try testing.expectEqual(Foo{ .b = "zigzag" }, buffer.get(1));
    try testing.expectEqual(Foo{ .b = "foobar" }, buffer.get(2));
}

test "sorting a span" {
    var buffer: MultiGapBuffer(struct { score: u32, chr: u8 }) = .{};
    defer buffer.deinit(testing.allocator);

    try buffer.ensureTotalCapacity(testing.allocator, 42);
    for (
        // zig fmt: off
        [42]u8{ 'b', 'a', 'c', 'a', 'b', 'c', 'b', 'c', 'b', 'a', 'b', 'a', 'b', 'c', 'b', 'a', 'a', 'c', 'c', 'a', 'c', 'b', 'a', 'c', 'a', 'b', 'b', 'c', 'c', 'b', 'a', 'b', 'a', 'b', 'c', 'b', 'a', 'a', 'c', 'c', 'a', 'c' },
        [42]u32{ 1,   1,   1,   2,   2,   2,   3,   3,   4,   3,   5,   4,   6,   4,   7,   5,   6,   5,   6,   7,   7,   8,   8,   8,   9,   9,  10,   9,  10,  11,  10,  12,  11,  13,  11,  14,  12,  13,  12,  13,  14,  14 },
        // zig fmt: on
    ) |chr, score| {
        buffer.appendBeforeAssumeCapacity(.{ .chr = chr, .score = score });
    }

    const sliced = buffer.slice();
    buffer.sortSpan(6, 21, struct {
        chars: []const u8,

        fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.chars[a] < ctx.chars[b];
        }
    }{ .chars = sliced.firstHalf(.chr) });

    var i: u32 = undefined;
    var j: u32 = 6;
    var c: u8 = 'a';

    while (j < 21) {
        i = j;
        j += 5;
        var n: u32 = 3;
        for (sliced.firstHalf(.chr)[i..j], sliced.firstHalf(.score)[i..j]) |chr, score| {
            try testing.expectEqual(score, n);
            try testing.expectEqual(chr, c);
            n += 1;
        }
        c += 1;
    }
}

test "0 sized struct field" {
    const a = testing.allocator;

    const Foo = struct {
        a: u0,
        b: f32,
    };

    var buffer = MultiGapBuffer(Foo){};
    defer buffer.deinit(a);

    try testing.expectEqualSlices(u0, &[_]u0{}, buffer.firstHalf(.a));
    try testing.expectEqualSlices(f32, &[_]f32{}, buffer.firstHalf(.b));

    try buffer.appendBefore(a, .{ .a = 0, .b = 42.0 });
    try testing.expectEqualSlices(u0, &[_]u0{0}, buffer.firstHalf(.a));
    try testing.expectEqualSlices(f32, &[_]f32{42.0}, buffer.firstHalf(.b));

    try buffer.insert(a, 0, .{ .a = 0, .b = -1.0 });
    buffer.moveGap(buffer.realLength());
    try testing.expectEqualSlices(u0, &[_]u0{ 0, 0 }, buffer.firstHalf(.a));
    try testing.expectEqualSlices(f32, &[_]f32{ -1.0, 42.0 }, buffer.firstHalf(.b));

    buffer.swapRemoveBefore(buffer.gap_end - 1);
    try testing.expectEqualSlices(u0, &[_]u0{0}, buffer.firstHalf(.a));
    try testing.expectEqualSlices(f32, &[_]f32{-1.0}, buffer.firstHalf(.b));
}

test "0 sized struct" {
    const a = testing.allocator;

    const Foo = struct {
        a: u0,
    };

    var buffer = MultiGapBuffer(Foo){};
    defer buffer.deinit(a);

    try testing.expectEqualSlices(u0, &[_]u0{}, buffer.firstHalf(.a));

    try buffer.appendBefore(a, .{ .a = 0 });
    try testing.expectEqualSlices(u0, &[_]u0{0}, buffer.firstHalf(.a));

    try buffer.insert(a, 0, .{ .a = 0 });
    buffer.moveGap(buffer.realLength());
    try testing.expectEqualSlices(u0, &[_]u0{ 0, 0 }, buffer.firstHalf(.a));

    buffer.swapRemoveBefore(buffer.gap_end - 1);
    try testing.expectEqualSlices(u0, &[_]u0{0}, buffer.firstHalf(.a));
}
