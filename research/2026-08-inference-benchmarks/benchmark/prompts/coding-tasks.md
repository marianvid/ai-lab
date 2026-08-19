# The ten coding tasks

Each prompt is sent alone, temperature 0. The answer's code block is
extracted and run against the hidden tests shown here, as user `nobody`,
with a 25-second timeout. Reference solutions that pass all ten are in
`../reference-solutions/`.

## t01_parse_duration

**Prompt**

> Write a Python function `parse_duration(s: str) -> int` that converts a duration string into whole seconds. It accepts a sequence of number+unit pairs with no separators, e.g. '1h30m', '45s', '2d4h', '90m'. Units: 'd' days, 'h' hours, 'm' minutes, 's' seconds. Whitespace anywhere must be ignored. Raise ValueError for an empty string, for a unit with no number, for an unknown unit, or for any trailing garbage. Return only the function in a single ```python code block, no explanation.

**Hidden tests**

```python
assert parse_duration('45s')==45
assert parse_duration('1h30m')==5400
assert parse_duration('2d4h')==187200
assert parse_duration('90m')==5400
assert parse_duration(' 1h 30 m ')==5400
assert parse_duration('0s')==0
for bad in ['','h','1x','1h30','abc','1h!']:
    try:
        parse_duration(bad); raise AssertionError('should have raised for %r'%bad)
    except ValueError: pass
```

## t02_merge_intervals

**Prompt**

> Write a Python function `merge_intervals(intervals)` taking a list of (start, end) integer tuples, possibly unsorted and possibly with start > end (treat those as invalid and raise ValueError). Return a new sorted list of merged, non-overlapping (start, end) tuples. Intervals that merely touch (e.g. (1,2) and (2,3)) must be merged. An empty input returns an empty list. Return only the function in a single ```python code block, no explanation.

**Hidden tests**

```python
assert merge_intervals([])==[]
assert merge_intervals([(1,3),(2,6),(8,10),(15,18)])==[(1,6),(8,10),(15,18)]
assert merge_intervals([(1,2),(2,3)])==[(1,3)]
assert merge_intervals([(5,7),(1,3)])==[(1,3),(5,7)]
assert merge_intervals([(1,10),(2,3)])==[(1,10)]
assert merge_intervals([(1,1)])==[(1,1)]
try:
    merge_intervals([(5,1)]); raise AssertionError('should raise')
except ValueError: pass
```

## t03_fix_bug

**Prompt**

> This function is meant to return the second-largest DISTINCT value in a list of numbers, or None when there is no such value. It is buggy. Return a corrected version.
> 
> ```python
> def second_largest(nums):
>     nums.sort()
>     return nums[-2]
> ```
> 
> It must not modify the caller's list. Return only the corrected function in a single ```python code block, no explanation.

**Hidden tests**

```python
orig=[3,1,4,1,5]
assert second_largest(orig)==4
assert orig==[3,1,4,1,5], 'caller list was mutated'
assert second_largest([2,2,2]) is None
assert second_largest([1]) is None
assert second_largest([]) is None
assert second_largest([5,3])==3
assert second_largest([-1,-2])==-2
```

## t04_group_by_key

**Prompt**

> Write a Python function `group_consecutive(items, key)` that walks a list and groups CONSECUTIVE items sharing the same key value (like itertools.groupby, but returning a concrete list). Return a list of (key_value, [items...]) tuples. Do not merge non-adjacent runs. An empty list returns an empty list. Return only the function in a single ```python code block, no explanation.

**Hidden tests**

```python
assert group_consecutive([], lambda x: x)==[]
assert group_consecutive([1,1,2,2,1], lambda x: x)==[(1,[1,1]),(2,[2,2]),(1,[1])]
r=group_consecutive(['aa','ab','ba'], lambda s: s[0])
assert r==[('a',['aa','ab']),('b',['ba'])], r
assert group_consecutive([5], lambda x:x)==[(5,[5])]
```

## t05_retry_decorator

**Prompt**

> Write a Python decorator factory `retry(times, exceptions=(Exception,))` that retries the wrapped function up to `times` TOTAL attempts when it raises one of `exceptions`. If all attempts fail, re-raise the last exception. Exceptions not in the tuple propagate immediately without retrying. The wrapper must preserve the function's __name__. No sleeping. Return only the code in a single ```python code block, no explanation.

**Hidden tests**

```python
calls={'n':0}
@retry(3)
def flaky():
    calls['n']+=1
    if calls['n']<3: raise ValueError('boom')
    return 'ok'
assert flaky()=='ok' and calls['n']==3

c2={'n':0}
@retry(2)
def always():
    c2['n']+=1
    raise KeyError('nope')
try:
    always(); raise AssertionError('should raise')
except KeyError: pass
assert c2['n']==2, c2

c3={'n':0}
@retry(3, exceptions=(ValueError,))
def wrong():
    c3['n']+=1
    raise TypeError('other')
try:
    wrong(); raise AssertionError('should raise')
except TypeError: pass
assert c3['n']==1, c3

@retry(2)
def named(): return 1
assert named.__name__=='named'
```

## t06_flatten

**Prompt**

> Write a Python function `flatten(obj, sep='.')` that flattens nested dicts and lists into a single-level dict of path -> scalar. Dict keys join with `sep`; list indices appear as numeric path segments. Example: {'a': {'b': [1, 2]}} becomes {'a.b.0': 1, 'a.b.1': 2}. An empty dict or empty list at a leaf position maps to itself as the value (e.g. {'a': {}} -> {'a': {}}). Return only the function in a single ```python code block, no explanation.

**Hidden tests**

```python
assert flatten({'a':{'b':[1,2]}})=={'a.b.0':1,'a.b.1':2}
assert flatten({'x':1})=={'x':1}
assert flatten({'a':{}})=={'a':{}}
assert flatten({'a':[]})=={'a':[]}
assert flatten({'a':{'b':{'c':3}}})=={'a.b.c':3}
r=flatten({'l':[{'k':1}]})
assert r=={'l.0.k':1}, r
assert flatten({'a':1},sep='/')=={'a':1}
assert flatten({'a':{'b':2}},sep='/')=={'a/b':2}
```

## t07_lru

**Prompt**

> Write a Python class `LRUCache` with `__init__(self, capacity)`, `get(key)` returning the value or None, and `put(key, value)`. It must evict the least-recently-used entry when over capacity. Both get and put count as a use. `len(cache)` must return the current number of entries. A capacity of 0 or less must raise ValueError. Return only the class in a single ```python code block, no explanation.

**Hidden tests**

```python
c=LRUCache(2)
c.put('a',1); c.put('b',2)
assert c.get('a')==1
c.put('c',3)
assert c.get('b') is None, 'b should have been evicted'
assert c.get('a')==1 and c.get('c')==3
assert len(c)==2
c.put('a',9)
assert c.get('a')==9 and len(c)==2
try:
    LRUCache(0); raise AssertionError('should raise')
except ValueError: pass
```

## t08_csv_quotes

**Prompt**

> Write a Python function `split_csv_line(line)` that splits ONE line of CSV into a list of field strings, without using the csv module. Rules: comma separates fields; a field may be wrapped in double quotes; inside quotes a comma is literal and a doubled quote ("") means one literal quote; unquoted fields are returned as-is including surrounding spaces. An unterminated quote raises ValueError. An empty line returns ['']. Return only the function in a single ```python code block, no explanation.

**Hidden tests**

```python
assert split_csv_line('a,b,c')==['a','b','c']
assert split_csv_line('')==['']
assert split_csv_line('a,,c')==['a','','c']
assert split_csv_line('"a,b",c')==['a,b','c']
assert split_csv_line('"he said ""hi""",x')==['he said "hi"','x']
assert split_csv_line(' a , b ')==[' a ',' b ']
try:
    split_csv_line('"abc'); raise AssertionError('should raise')
except ValueError: pass
```

## t09_topo

**Prompt**

> Write a Python function `topo_sort(graph)` where graph is a dict mapping node -> list of nodes it depends on. Return a list of all nodes ordered so every node appears after its dependencies. Among nodes that are equally free, pick the smallest by natural sort order so the result is deterministic. Raise ValueError('cycle') if the graph contains a cycle. Nodes that appear only as dependencies must still be included. Return only the function in a single ```python code block, no explanation.

**Hidden tests**

```python
assert topo_sort({})==[]
r=topo_sort({'b':['a'],'a':[]})
assert r==['a','b'], r
r=topo_sort({'c':['a','b'],'b':['a'],'a':[]})
assert r==['a','b','c'], r
r=topo_sort({'x':['z']})
assert r==['z','x'], r
r=topo_sort({'b':[],'a':[]})
assert r==['a','b'], r
try:
    topo_sort({'a':['b'],'b':['a']}); raise AssertionError('should raise')
except ValueError: pass
```

## t10_semver

**Prompt**

> Write a Python function `compare_versions(a, b)` returning -1, 0 or 1 comparing two semantic version strings like '1.2.3', '1.2.3-alpha.1', '2.0.0-rc.2+build5'. Rules: compare major/minor/patch numerically; a version WITH a pre-release is lower than the same version without one; pre-release identifiers compare left to right, numeric identifiers numerically and lower than alphanumeric ones, alphanumeric ones ASCII-wise; build metadata after '+' is ignored entirely. Raise ValueError on a malformed version. Return only the function in a single ```python code block, no explanation.

**Hidden tests**

```python
assert compare_versions('1.2.3','1.2.3')==0
assert compare_versions('1.2.4','1.2.3')==1
assert compare_versions('1.10.0','1.9.0')==1
assert compare_versions('1.2.3-alpha','1.2.3')==-1
assert compare_versions('1.2.3','1.2.3-alpha')==1
assert compare_versions('1.2.3-alpha.1','1.2.3-alpha.2')==-1
assert compare_versions('1.2.3-1','1.2.3-alpha')==-1
assert compare_versions('1.2.3+b1','1.2.3+b2')==0
assert compare_versions('2.0.0-rc.2+build5','2.0.0')==-1
for bad in ['1.2','x.y.z','1.2.3.4','']:
    try:
        compare_versions(bad,'1.2.3'); raise AssertionError('should raise for %r'%bad)
    except ValueError: pass
```

