import std.typetuple, std.parallelism, std.range, std.functional,
    std.algorithm, std.stdio, std.array, std.traits, std.conv,
    core.stdc.string, core.atomic;

version(unittest) {
    import std.random, std.typecons, std.math;
}

private template finiteRandom(R) {
    enum bool finiteRandom = isRandomAccessRange!R && std.range.hasLength!R;
}

// Tracks whether the last merge was from -> buf or buf -> from.  This
// avoids needing to copy from buf to from after every iteration.
private enum MergedTo {
    from,
    buf
}

/**
Sort a range using a parallel merge sort algorithm, falling back to
$(D baseAlgorithm) for small subranges.  Usage is similar to
$(XREF algorithm, sort).

Params:

pred = The predicate to sort on.

baseAlgorithm = The algorithm to fall back to for small subranges.
$(D parallelSort) is a stable sort iff $(D baseAlgorithm) is a stable sort.

range = The range to be sorted.

minParallelSort = The smallest subrange to sort in parallel. Small values
will expose more parallelism, but also incur more overhead.

minParallelMerge = The smallest subrange to merge in parallel.  Since
merging is a cheaper operation than sorting, this should be somewhat larger
than $(D minParallelSort).

pool = The $(XREF parallelism, TaskPool) to use.  If null, the global
default task pool returned by $(XREF parallelism, taskPool) will be used.
*/
SortedRange!(R, pred)
parallelSort(alias pred = "a < b", alias baseAlgorithm = std.algorithm.sort, R)(
    R range,
    size_t minParallelSort = 1024,
    size_t minParallelMerge = 4096,
    TaskPool pool = null
) if(finiteRandom!R && hasAssignableElements!R) {
    // TODO:  Use C heap or TempAlloc or something.
    auto buf = new ElementType!(R)[range.length];

    if(pool is null) pool = std.parallelism.taskPool;

    immutable mergedTo = parallelSortImpl!(pred, baseAlgorithm, R, typeof(buf))
        (range, buf, minParallelSort, minParallelMerge, pool
    );

    if(mergedTo == MergedTo.buf) {
        copy(buf, range);
    }

    return SortedRange!(R, pred)(range);
}

unittest {
    // This algorithm is kind of complicated with all the tricks to prevent
    // excess copying and stuff.  Use monte carlo unit testing.

    auto gen = Random(314159265);  // Make tests deterministic but pseudo-random.
    foreach(i; 0..100) {
        auto nums = new uint[uniform(10, 20, gen)];
        foreach(ref num; nums) {
            num = uniform(0, 1_000_000, gen);
        }

        auto duped = nums.dup;
        parallelSort!"a > b"(duped, 4, 8);
        sort!"a > b"(nums);
        assert(duped == nums);
    }

    // Test sort stability.
    auto arr = new Tuple!(int, int)[32_768];
    foreach(ref elem; arr) {
        elem[0] = uniform(0, 10, gen);
        elem[1] = uniform(0, 10, gen);
    }

    static void stableSort(alias pred, R)(R range) {
        // Quick and dirty insertion sort, for testing only.
        alias binaryFun!pred comp;

        foreach(i; 1..range.length) {
            for(size_t j = i; j > 0; j--) {
                if(comp(range[j], range[j - 1])) {
                    swap(range[j], range[j - 1]);
                } else {
                    break;
                }
            }
        }
    }

    parallelSort!("a[1] < b[1]", stableSort)(arr);
    assert(isSorted!"a[1] < b[1]"(arr));
    parallelSort!("a[0] < b[0]", stableSort)(arr);
    assert(isSorted!"a[0] < b[0]"(arr));

    foreach(i; 0..arr.length - 1) {
        if(arr[i][0] == arr[i + 1][0]) {
            assert(arr[i][1] <= arr[i + 1][1]);
        }
    }
}

MergedTo parallelSortImpl(alias pred, alias baseAlgorithm, R1, R2)(
    R1 range,
    R2 buf,
    size_t minParallelSort,
    size_t minParallelMerge,
    TaskPool pool
) {
    assert(pool);

    if(range.length < minParallelSort) {
        baseAlgorithm!pred(range);
        return MergedTo.from;
    }

    immutable len = range.length;
    auto left = range[0..len / 2];
    auto right = range[len / 2..len];
    auto bufLeft = buf[0..len / 2];
    auto bufRight = buf[len / 2..len];

    auto ltask = scopedTask!(parallelSortImpl!(pred, baseAlgorithm, R1, R2))(
        left, bufLeft, minParallelSort, minParallelMerge, pool
    );
    pool.put(ltask);

    immutable rloc = parallelSortImpl!(pred, baseAlgorithm, R1, R2)(
        right, bufRight, minParallelSort, minParallelMerge, pool
    );

    auto lloc = ltask.yieldForce();

    if(lloc == MergedTo.from && rloc == MergedTo.buf) {
        copy(left, bufLeft);
        lloc = MergedTo.buf;
    } else if(lloc == MergedTo.buf && rloc == MergedTo.from) {
        copy(right, bufRight);
    }

    if(lloc == MergedTo.from) {
        parallelMerge!(pred, R1, R1, R2)(left, right, buf, minParallelMerge);
        return MergedTo.buf;
    } else {
        parallelMerge!(pred, R2, R2, R1)(bufLeft, bufRight, range, minParallelMerge);
        return MergedTo.from;
    }
}

/**
Merge ranges $(D from1) and $(D from2), which are assumed sorted according
to $(D pred), into $(D buf) using a parallel divide-and-conquer algorithm.

Params:

from1 = The first of the two sorted ranges to be merged.  This must be a
random access range with length.

from2 = The second of the two sorted ranges to be merged.  This must also
be a random access range with length and must have an identical element type to
$(D from1).

buf = The buffer to merge into.  This must be a random access range with
length equal to $(D from1.length + from2.length) and must have assignable
elements.

minParallel = The minimum merge size to parallelize.  Smaller values
create more parallel work units resulting in greater scalability but
increased overhead.

pool = The $(XREF parallelism, TaskPool) to use.  If null, the global
default task pool returned by $(XREF parallelism, taskPool) will be used.
*/
void parallelMerge(alias pred = "a < b", R1, R2, R3)(
    R1 from1,
    R2 from2,
    R3 buf,
    size_t minParallel = 4096,
    TaskPool pool = null
) if(allSatisfy!(finiteRandom, TypeTuple!(R1, R2, R3)) &&
   is(ElementType!R1 == ElementType!R2) &&
   is(ElementType!R2 == ElementType!R3) &&
   hasAssignableElements!R3
)
in {
    assert(from1.length + from2.length == buf.length);
} body {
    if(buf.length < minParallel) {
        return merge!(pred, R1, R2, R3)(from1, from2, buf);
    }

    immutable len1 = from1.length;
    immutable len2 = from2.length;

    if(len1 == 0 && len2 == 0) {
        return;
    }

    typeof(from1) left1, right1;
    typeof(from2) left2, right2;
    alias binaryFun!pred comp;

    if(len1 > len2) {
        auto mid1Index = len1 / 2;

        // This is necessary to make the sort stable:
        while(mid1Index > 0 && !comp(from1[mid1Index - 1], from1[mid1Index])) {
            mid1Index--;
        }

        auto mid1 = from1[mid1Index];
        left1 = from1[0..mid1Index];
        right1 = from1[mid1Index..len1];
        left2 = assumeSorted!pred(from2).lowerBound(mid1).release;
        right2 = from2[left2.length..len2];
    } else {
        auto mid2Index = len2 / 2;

        // This is necessary to make the sort stable:
        while(mid2Index > 0 && !comp(from2[mid2Index - 1], from2[mid2Index])) {
            mid2Index--;
        }

        auto mid2 = from2[mid2Index];
        left2 = from2[0..mid2Index];
        right2 = from2[mid2Index..len2];
        left1 = assumeSorted!pred(from1).lowerBound(mid2).release;
        right1 = from1[left1.length..len1];
    }

    auto leftBuf = buf[0..left1.length + left2.length];
    auto rightBuf = buf[leftBuf.length..buf.length];

    if(leftBuf.length == 0 || rightBuf.length == 0) {
        // Then recursing further would lead to infinite recursion.
        return merge!(pred, R1, R2, R3)(from1, from2, buf);
    }

    if(pool is null) pool = std.parallelism.taskPool;

    auto rightTask = scopedTask!(parallelMerge!(pred, R1, R2, R3))(
        right1, right2, rightBuf, minParallel, pool
    );

    pool.put(rightTask);
    parallelMerge!(pred, R1, R2, R3)(left1, left2, leftBuf, minParallel, pool);
    rightTask.yieldForce();
}

unittest {
    auto from1 = [1, 2, 4, 8, 16, 32];
    auto from2 = [2, 4, 6, 8, 10, 12];
    auto buf = new int[from1.length + from2.length];
    parallelMerge(from1, from2, buf, 2);
    assert(buf == [1, 2, 2, 4, 4, 6, 8, 8, 10, 12, 16, 32]);
}

/**
Merge ranges $(D from1) and $(D from2), which are assumed sorted according
to $(D pred), into $(D buf) using a sequential algorithm.

Params:

from1 = The first of the two sorted ranges to be merged.

from2 = The second of the two sorted ranges to be merged.  This must also
be an input range and must have an identical element type to
$(D from1).

buf = The buffer to merge into.  This must be an output range with
capacity at least equal to $(D walkLength(from1) + walkLength(from2)).

Example:
---
auto from1 = [1, 2, 4, 8, 16, 32];
auto from2 = [2, 4, 6, 8, 10, 12];
auto buf = new int[from1.length + from2.length];
merge(from1, from2, buf);
assert(buf == [1, 2, 2, 4, 4, 6, 8, 8, 10, 12, 16, 32]);
---
*/
void merge(alias pred = "a < b", R1, R2, R3)(
    R1 from1,
    R2 from2,
    R3 buf
) if(allSatisfy!(isInputRange, TypeTuple!(R1, R2)) &&
     is(ElementType!R1 == ElementType!R2) &&
     is(ElementType!R2 == ElementType!R3) &&
     isOutputRange!(R3, ElementType!R1)
) {
    alias binaryFun!(pred) comp;

    static if(allSatisfy!(isRandomAccessRange, TypeTuple!(R1, R2, R3))) {
        // This code is empirically slightly more efficient in the case of
        // arrays.
        size_t index1 = 0, index2 = 0, bufIndex = 0;
        immutable len1 = from1.length;
        immutable len2 = from2.length;

        while(index1 < len1 && index2 < len2) {
            if(comp(from2[index2], from1[index1])) {
                buf[bufIndex] = from2[index2];
                index2++;
            } else {
                buf[bufIndex] = from1[index1];
                index1++;
            }

            bufIndex++;
        }

        if(index1 < len1) {
            assert(index2 == len2);
            copy(from1[index1..len1], buf[bufIndex..len1 + len2]);
        } else if(index2 < len2) {
            assert(index1 == len1);
            copy(from2[index2..len2], buf[bufIndex..len1 + len2]);
        }
    } else {
        // Fall back to the obvious generic impl.
        while(!from1.empty && !from2.empty) {
            if(comp(from2.front, from1.front)) {
                buf.put(from2.front);
                from2.popFront();
            } else {
                buf.put(from1.front);
                from1.popFront();
            }
        }

        if(!from1.empty) {
            assert(from2.empty);
            copy(from1, buf);
        } else if(!from2.empty) {
            assert(from1.empty);
            copy(from2, buf);
        }
    }
}

unittest {
    auto from1 = [1, 2, 4, 8, 16, 32];
    auto from2 = [2, 4, 6, 8, 10, 12];
    auto buf = new int[from1.length + from2.length];
    merge(from1, from2, buf);
    assert(buf == [1, 2, 2, 4, 4, 6, 8, 8, 10, 12, 16, 32]);
}

void parallelMergeInPlace(alias pred = "a < b", R)(
    R range, 
    size_t middle,
    size_t minParallel = 1024,
    TaskPool pool = null
) {
    if(pool is null) pool = taskPool;
    immutable rlen = range.length;    
    alias binaryFun!(pred) comp;

    static size_t largestLess(T)(T[] data, T value) {
        return assumeSorted!(comp)(data).lowerBound(value).length;
    }

    static size_t smallestGr(T)(T[] data, T value) {
        return data.length - 
            assumeSorted!(comp)(data).upperBound(value).length;
    }

    if (range.length < 2 || middle == 0 || middle == range.length) {
        return;
    }
    
    if (range.length == 2) {
        if(comp(range[1], range[0])) {
            swap(range[0], range[1]);
        }
        
        return;
    }

    size_t half1, half2;

    if (middle > range.length - middle) {
        half1 = middle / 2;
        auto pivot = range[half1];
        half2 = largestLess(range[middle..rlen], pivot);
    } else {
        half2 = (range.length - middle) / 2;
        auto pivot = range[half2 + middle];
        half1 = smallestGr(range[0..middle], pivot);
    }

    bringToFront(range[half1..middle], range[middle..middle + half2]);
    size_t newMiddle = half1 + half2;

    auto left = range[0..newMiddle];
    auto right = range[newMiddle..range.length];

    if(left.length >= minParallel) {
        auto leftTask = scopedTask!(parallelMergeInPlace!(pred, R))
            (left, half1, minParallel, pool);
        taskPool.put(leftTask);
        parallelMergeInPlace!(pred, R)
            (right, half2 + middle - newMiddle, minParallel, pool);
        leftTask.yieldForce();
    } else {
        parallelMergeInPlace!(pred, R)(left, half1, minParallel, pool);
        parallelMergeInPlace!(pred, R)
            (right, half2 + middle - newMiddle, minParallel, pool);
    }
}

unittest {
    auto arr = new int[10_000];
    
    // Make sure serial and parallel both work by bypassing parallelism
    // by making minParallel huge.
    foreach(minParallel; [64, 20_000]) {
        copy(iota(0, 10_000, 2), arr[0..5_000]);
        copy(iota(1, 10_000, 2), arr[5_000..$]);
        parallelMergeInPlace(arr, 5_000, minParallel);
        assert(equal(arr, iota(10_000)), to!string(minParallel));
    }
}         

// In a few implementations we need to create custom ranges to be reduced.
// std.parallelism.reduce checks for a random access range to conform to
// Phobos conventions but only actually uses opIndex and length.
// This mixin adds stubs of the other primitives to pass
// isRandomAccessRange.
//
// This returns a string instead of using mixin templates to get around some
// weird forward referencing issues.
private template dummyRangePrimitives(string elemType) {
    enum string dummyRangePrimitives =
    elemType ~ q{ front() @property { assert(0); }
    } ~ elemType ~ q{ back() @property { assert(0); }
    void popFront() { assert(0); }
    void popBack() @property { assert(0); }
    typeof(this) opSlice(size_t foo, size_t bar) { assert(0); }
    typeof(this) save() { assert(0); }
    bool empty() @property { assert(0); }
    };
}

CommonType!(ElementType!(Range1),ElementType!(Range2))
parallelDotProduct(Range1, Range2)(
    Range1 a,
    Range2 b,
    TaskPool pool = null,
    size_t workUnitSize = size_t.max
) if(isFloatingPoint!(ElementType!Range1) && isFloatingPoint!(ElementType!Range2)
   && isRandomAccessRange!Range1 && isRandomAccessRange!Range2
   && hasLength!Range1 && hasLength!Range2
) in {
    assert(a.length == b.length);
} body {

    if(pool is null) pool = taskPool;
    if(workUnitSize == size_t.max) {
        workUnitSize = pool.defaultWorkUnitSize(a.length);
    }

    alias typeof(return) F;

    // A TaskPool.reduce/std.algorithm.map approach seems like the obvious one
    // here, but std.numeric.dotProduct is so well optimized that I can't beat
    // it unless I use it under the hood.  Create a range of dot products of
    // slices, and let TaskPool.reduce handle the minutiae of sending each
    // element to the task pool.  Eventually this slice-reduce paradigm
    // needs to be extracted into reusable code.
    static struct SliceDotProd {
        Range1 a;
        Range2 b;
        size_t workUnitSize;

        mixin(dummyRangePrimitives!"F");

        size_t length() @property {
            assert(a.length == b.length);
            return (a.length / workUnitSize) + (a.length % workUnitSize > 0);
        }

        F opIndex(size_t index) {
            immutable start = workUnitSize * index;
            immutable stop = min(a.length, workUnitSize * (index + 1));
            assert(start <= stop, text(start, ' ', stop));
            return std.numeric.dotProduct(a[start..stop], b[start..stop]);
        }
    }

    return taskPool.reduce!"a + b"(cast(F) 0, SliceDotProd(a, b, workUnitSize), 1);
}

unittest {
    auto a = new double[10_000];
    auto b = new double[10_000];

    foreach(i, ref elem; a) {
        elem = i;
        b[i] = i;
    }

    auto serial = std.numeric.dotProduct(a, b);
    auto parallel = parallelDotProduct(a, b, taskPool);
    assert(approxEqual(serial, parallel), text(serial, ' ', parallel));
}

size_t parallelCount(alias pred = "a == b", Range, E)
(Range r, E value, TaskPool pool = null, size_t workUnitSize = size_t.max)
if(isRandomAccessRange!Range && hasLength!Range) {

    if(pool is null) pool = taskPool;

    if(workUnitSize == size_t.max) {
        workUnitSize = pool.defaultWorkUnitSize(r.length);
    }

    static struct MapPred {
        Range r;
        E value;

        mixin(dummyRangePrimitives!"size_t");

        size_t length() @property {
            return r.length;
        }

        size_t opIndex(size_t index) {
            return (binaryFun!pred(r[index], value)) ? 1 : 0;
        }
    }

    return taskPool.reduce!"a + b"(
        cast(size_t) 0, MapPred(r, value), workUnitSize
    );
}

size_t parallelCount(alias pred = "true", Range)
(Range r, TaskPool pool = null, size_t workUnitSize = size_t.max)
if(isRandomAccessRange!Range && hasLength!Range) {

    if(pool is null) pool = taskPool;

    if(workUnitSize == size_t.max) {
        workUnitSize = pool.defaultWorkUnitSize(r.length);
    }

    static size_t predToSizeT(T)(T val) {
        return (unaryFun!pred(val)) ? 1 : 0;
    }

    return taskPool.reduce!"a + b"(cast(size_t) 0,
        std.algorithm.map!predToSizeT(r), workUnitSize
    );
}

unittest {
    assert(parallelCount([1, 2, 1, 2, 3], 2) == 2);
    assert(parallelCount!"a == 2"([1, 2, 1, 2, 3]) == 2);
}

void parallelAdjacentDifference(alias pred = "a - b", R1, R2)(
    R1 input,
    R2 output,
    TaskPool pool = null,
    size_t workUnitSize = size_t.max
) if(allSatisfy!(isRandomAccessRange, TypeTuple!(R1, R2)) &&
   hasAssignableElements!(R2) &&
   is(typeof(binaryFun!pred(R1.init[0], R1.init[1])) : ElementType!R2)
) {
    static size_t getLength(R)(ref R range) {
        static if(is(typeof(range.length) : size_t)) {
            return range.length;
        } else {
            return size_t.max;
        }
    }

    // getLength(output) + 1 because we need one less element in output
    // than we had in input.
    immutable minLen = min(getLength(input), getLength(output) + 1);
    if(pool is null) pool = taskPool;

    if(workUnitSize == size_t.max) {
        workUnitSize = pool.defaultWorkUnitSize(minLen - 1);
    }

    // Using parallel foreach to iterate over individual elements is too
    // slow b/c of delegate overhead for such fine grained parallelism.
    // Use parallel foreach to iterate over slices and handle them serially.
    auto sliceStarts = iota(0, minLen - 1, workUnitSize);

    foreach(startIndex; pool.parallel(sliceStarts, 1)) {
        immutable endIndex = min(startIndex + workUnitSize, minLen - 1);

        // This avoids some indirection and seems to be faster.
        auto ip = input;
        auto op = output;

        foreach(i; startIndex..endIndex) {
            op[i] = binaryFun!pred(ip[i + 1], ip[i]);
        }
    }
}

unittest {
    auto input = [1, 2, 4, 8, 16, 32];
    auto output = new int[5];
    parallelAdjacentDifference(input, output);
    assert(output == [1, 2, 4, 8, 16]);
}

bool parallelEqual(alias pred = "a == b", R1, R2)(
    R1 range1, 
    R2 range2,
    size_t workUnitSize = size_t.max,
    TaskPool pool = null
) 
if(isRandomAccessRange!R1 && isRandomAccessRange!R2 &&
hasLength!R1 && hasLength!R2) {
    
    if(range1.length != range2.length) return false;
    immutable len = range1.length;
    
    if(pool is null) pool = taskPool;
    if(workUnitSize == size_t.max) {
        workUnitSize = pool.defaultWorkUnitSize(len);
    }
    
    auto chunks1 = std.range.chunks(range1, workUnitSize);   
    auto chunks2 = std.range.chunks(range2, workUnitSize);   
    assert(chunks1.length == chunks2.length);
    immutable nChunks = chunks1.length;
    
    bool ret = true;
    
    try {
        foreach(chunkIndex; pool.parallel(iota(nChunks), 1)) {
            auto c1 = chunks1[chunkIndex];
            auto c2 = chunks2[chunkIndex];
            if(!std.algorithm.equal!pred(c1, c2)) {
                atomicStore(ret, false);
                break;
            }
        }
    } catch(ParallelForeachError) {
        // Ignore it.  It's because we tried to break out of a parallel
        // foreach loop. 
    }
    
    return ret;
}    

unittest {
    assert(parallelEqual([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]));
    assert(!parallelEqual([1, 2, 3, 4, 5, 6], [1, 3, 3, 4, 5, 6]));
    assert(!parallelEqual([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7]));
}

//////////////////////////////////////////////////////////////////////////////
// Benchmarks
//////////////////////////////////////////////////////////////////////////////
import std.random, std.datetime, std.exception, std.numeric;

void mergeBenchmark() {
    enum N = 8192;
    enum nIter = 1000;
    auto a = new float[N];
    auto b = new float[N];
    auto buf = new float[a.length + b.length];

    foreach(ref elem; chain(a, b)) elem = uniform(0f, 1f);
    sort(a);
    sort(b);

    auto sw = StopWatch(AutoStart.yes);
    foreach(i; 0..nIter) merge(a, b, buf);
    writeln("Serial Merge:  ", sw.peek.msecs);
    assert(equal(buf, sort(a ~ b)));

    sw.reset();
    foreach(i; 0..nIter) parallelMerge(a, b, buf, 2048);
    writeln("Parallel Merge:  ", sw.peek.msecs);
    assert(equal(buf, sort(a ~ b)));
}

void mergeInPlaceBenchmark() {
    enum N = 8192;
    enum nIter = 100;
    auto ab = new float[2 * N];
    auto a = ab[0..$ / 2];
    auto b = ab[$ / 2..$];

    foreach(ref elem; ab) elem = uniform(0f, 1f);

    auto sw = StopWatch(AutoStart.no);
    foreach(i; 0..nIter) {
        randomShuffle(ab);
        sort(a);
        sort(b);
        sw.start();
        
        // Disable parallelism by setting minParallel to be bigger than N.
        parallelMergeInPlace(ab, N / 2, N + 1);
        sw.stop();
    }
    writeln("Serial In-place Merge:  ", sw.peek.msecs);

    sw.reset();
    foreach(i; 0..nIter) {
        randomShuffle(ab);
        sort(a);
        sort(b);
        sw.start();
        
        // Use default minParallel.
        parallelMergeInPlace(ab, N / 2);
        sw.stop();
    }
    writeln("Parallel In-place Merge:  ", sw.peek.msecs);
}

void sortBenchmark() {
    enum N = 32768;
    enum nIter = 100;
    auto a = new ushort[N];
    foreach(ref elem; a) elem = uniform(cast(ushort) 0, ushort.max);

    auto sw = StopWatch(AutoStart.yes);
    sort(a);
    writeln("Serial Sort:    ", sw.peek.usecs);
    enforce(isSorted(a));

    randomShuffle(a);
    sw.reset();
    parallelSort!("a < b", sort)(a, 4096, 4096);
    writeln("Parallel Sort:  ", sw.peek.usecs);
    enforce(isSorted(a));
}

void dotProdBenchmark() {
    enum n = 100_000;
    enum nIter = 100;
    auto a = new float[n];
    auto b = new float[n];

    foreach(ref num; chain(a, b)) {
        num = uniform(0.0, 1.0);
    }

    auto sw = StopWatch(AutoStart.yes);
    foreach(i; 0..nIter) std.numeric.dotProduct(a, b);
    writeln("Serial dot product:    ", sw.peek.msecs);

    sw.reset();
    foreach(i; 0..nIter) parallelDotProduct(a, b, taskPool);
    writeln("Parallel dot product:  ", sw.peek.msecs);
}

void countBenchmark() {
    enum n = 3_000;
    enum nIter = 1_000;
    auto nums = new int[n];
    foreach(ref elem; nums) elem = uniform(0, 3);

    auto sw = StopWatch(AutoStart.yes);
    foreach(i; 0..nIter) count(nums, 2);
    writeln("Serial count by value:    ", sw.peek.msecs);

    sw.reset();
    foreach(i; 0..nIter) parallelCount(nums, 2);
    writeln("Parallel count by value:  ", sw.peek.msecs);

    sw.reset();
    foreach(i; 0..nIter) count!"a == 2"(nums);
    writeln("Serial count by pred:    ", sw.peek.msecs);

    sw.reset();
    foreach(i; 0..nIter) parallelCount!"a == 2"(nums);
    writeln("Parallel count by pred:  ", sw.peek.msecs);
}

void adjacentDifferenceBenchmark() {
    enum n = 500_000;
    enum nIter = 100;

    auto input = new int[n];
    auto output = new int[n - 1];
    foreach(ref elem; input) {
        elem = uniform(0, 10_000);
    }

    auto sw = StopWatch(AutoStart.yes);
    foreach(iter; 0..nIter) {
        // Quick n' dirty inline serial impl.
        foreach(i; 0..n - 1) {
            output[i] = input[i + 1] - input[i];
        }
    }

    writeln("Serial adjacent difference:  ", sw.peek.msecs);

    sw.reset();
    foreach(iter; 0..nIter) parallelAdjacentDifference(input, output);
    writeln("Parallel adjacent difference:  ", sw.peek.msecs);
}

void equalBenchmark() {
    enum n = 20_000;
    enum nIter = 1_000;
    
    auto a = new int[n];
    auto b = new int[n];
    
    auto sw = StopWatch(AutoStart.yes);
    foreach(i; 0..nIter) equal(a, b);
    writeln("Serial equal:  ", sw.peek.msecs);
    
    sw.reset();
    foreach(i; 0..nIter) parallelEqual(a, b);
    writeln("Parallel equal:  ", sw.peek.msecs);
}

void main() {
    mergeBenchmark();
    mergeInPlaceBenchmark();
    sortBenchmark();
    dotProdBenchmark();
    countBenchmark();
    adjacentDifferenceBenchmark();
    equalBenchmark();
}
