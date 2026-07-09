# Python Recall: The Concepts You Always Forget

This is not a complete Python course. It deliberately skips the basics you already know and skips OOP entirely, since that deserves its own separate note. What this blog covers is the exact opposite of a full course: the specific concepts that are small enough to get overlooked while studying, but confusing enough to trip you up the moment you actually hit them in real code.

Each topic below is written as a question you might genuinely ask yourself while coding, followed by a clear answer, and then a code example with its output, so you see exactly what happens rather than just reading a description of it.

## Part 1: How Python Variables Actually Work Behind the Scenes

Almost every confusing Python behavior later in this note traces back to one core idea, so it is worth understanding first, properly.

### 1. Are Python variables just boxes that store a value, like in C?

No, and this single realization changes how a lot of other Python behavior makes sense. In a language like C, a variable is a labeled box with its own memory address, so two variables holding the same value still live in two different places. Python variables do not work this way. A Python variable is simply a name that points to an object sitting somewhere in memory. If two variables point to the same object, they are not just equal, they are literally referring to the exact same thing in memory.

```python
a = 10   # a now points to an integer object with value 10
b = 10   # b happens to point to the exact same integer object

print(id(a))   # prints the memory address a points to
print(id(b))   # prints the memory address b points to
```

Output:

```
140719327894120
140719327894120
```

Both `id()` values are identical. Python is not creating two separate integer objects here. Since small integers are immutable, meaning they can never be changed after creation, Python safely lets multiple variables point to the very same object instead of wasting memory creating duplicates. This single idea is the key to understanding mutable versus immutable objects, how assignment behaves, how shallow and deep copies differ, and how function arguments get passed, all of which show up again later in this note.

### 2. If strings can be reassigned, why can't I change a few characters inside one?

This confuses a lot of people, since both of these look like they should be doing the same kind of thing.

```python
text = "telus"
text = "TELus"   # this works with no error at all
```

```python
text = "telus"
text[0:3] = "TEL"   # this raises an error
```

```
TypeError: 'str' object does not support item assignment
```

Here is the key difference. In the first example, you are not changing the string `"telus"` at all. You are simply making the name `text` point to a brand new string object, `"TELus"`, while the original string still exists unchanged, just with nothing pointing to it anymore. In the second example, you are trying to reach inside the existing string object and directly overwrite a few of its characters, and Python refuses, because strings are immutable. Once a string object is created, its actual contents can never be changed, no matter what you do to it.

The rule to hold onto: variables are free to point to something new whenever you want, but immutable objects themselves can never be edited in place.

### 3. Why did clearing my temp list also wipe out the value I already saved in ans?

This is one of the most common "Python broke my code for no reason" moments, and it comes directly from the reference idea covered above.

```python
temp = [1, 2]
ans = []

ans.append(temp)   # ans now holds a reference to the SAME list object as temp
temp.clear()        # clearing temp also empties the list ans is pointing to

print(ans)
```

```
[[]]
```

The confusion usually sounds like this: "I only cleared temp, so ans should still have `[1, 2]` safely stored inside it." But `append()` does not store a copy of `temp`, it stores a reference to the exact same list object that `temp` also points to. So when `temp.clear()` empties that list, `ans` is still pointing at that same now-empty list, which is why you see `[[]]` instead of `[[1, 2]]`.

The fix is to explicitly store a separate copy instead of a reference to the original.

```python
ans.append(temp.copy())   # stores an independent copy, not the same object
# or equivalently
ans.append(temp[:])        # slicing an entire list also creates a copy
```

### 4. How does a, b = b, a swap two variables without a temporary variable?

The first time you see this, it genuinely looks like Python is breaking the normal rules of assignment.

```python
a = 5
b = 6

a, b = b, a

print(a, b)
```

```
6 5
```

The trick is that Python does not execute this assignment strictly left to right the way you might assume. Instead, it first fully evaluates the entire right hand side, `b, a`, which packs those two values into a temporary tuple, `(6, 5)`, before any assignment happens at all. Only after that temporary tuple is safely built does Python unpack it back into `a` and `b`.

Conceptually, Python is doing something close to this internally.

```python
temp = (b, a)   # tuple packing happens first, capturing both original values safely
a, b = temp      # tuple unpacking happens second, assigning both at once
```

Since both original values are safely tucked into that temporary tuple before either `a` or `b` gets overwritten, nothing is ever lost. This is exactly why the swap works with zero risk of clobbering a value early.

## Part 2: Lists You Need to Actually Understand

### 5. What is list comprehension and why should I use it instead of a loop with append?

List comprehension is a compact way to build a brand new list directly from an existing iterable, in a single readable line, instead of writing a full loop with a manual `append()` call inside it.

Syntax:

```python
new_list = [expression for item in iterable if condition]
```

Basic example:

```python
nums = [1, 2, 3, 4]

square = [i * i for i in nums]   # square every number in nums

print(square)
```

```
[1, 4, 9, 16]
```

Adding a condition filters which items get included at all.

```python
even = [i for i in nums if i % 2 == 0]   # keep only even numbers

print(even)
```

```
[2, 4]
```

A few things worth remembering about list comprehension. It always returns a brand new list. The original iterable you looped over is left completely unchanged. And it directly replaces the common pattern of writing an empty list, looping, and calling `append()` inside the loop, in a single more concise line.

### 6. What are the most common list comprehension mistakes?

Two small mistakes come up again and again, and both are worth calling out directly since they look almost correct at a glance.

The first is putting the brackets in the wrong place.

```python
# Wrong: the brackets only wrap "i + sum", the for loop is left outside them
l = [i + sum] for i in list1
```

The correct version keeps the entire expression, including the loop, inside one set of brackets.

```python
total = 0
list1 = [1, 2, 3]

l = [i + total for i in list1]   # correct: for loop lives inside the brackets

print(l)
```

```
[1, 2, 3]
```

The general form to always remember is `[expression for item in iterable]`, all as one contained unit.

The second mistake is using `sum` as a variable name.

```python
# Wrong: this silently overrides Python's own built-in sum() function
sum = 0
```

The problem is that `sum` is already a built-in Python function used to add up the values inside an iterable, `sum([1, 2, 3])`. The moment you assign `sum = 0`, you overwrite that built-in inside your current scope, and any later call to the real `sum()` function will break or behave unexpectedly. The simple fix is to just use a different name.

```python
total = 0   # use "total" instead of "sum" to avoid overriding the built-in
```

### 7. How does indexing work when a list contains other lists inside it?

Nested list indexing confuses people mainly because it looks like you need to do something clever, when really you just apply one index at a time, working from the outside in.

```python
list1 = [10, 20, [300, 400, [5000, 6000], 500], 30]
```

Suppose you want to reach the value `400`.

```python
list1[2][1]   # step 1: list1[2] gets the inner list, step 2: [1] gets 400 from it
```

```
400
```

Breaking it down explicitly: `list1[2]` first gives you `[300, 400, [5000, 6000], 500]`, since that is the item sitting at index 2 of the outer list. Then applying `[1]` to that result gives you `400`, since that is the item sitting at index 1 inside that inner list. Reading nested indexing left to right, one step at a time, removes almost all the confusion around it.

### 8. What is the difference between append() and insert()?

Both add a value to a list, but they differ in exactly where that value ends up.

`append()` always adds the new value at the very end of the list, with no choice about position.

```python
nums = [1, 2]
nums.append(3)   # adds 3 at the end

print(nums)
```

```
[1, 2, 3]
```

`insert()` lets you choose exactly where the new value goes, by giving it a specific index position first.

```python
nums = [1, 2, 4]
nums.insert(2, 3)   # insert the value 3 at index position 2

print(nums)
```

```
[1, 2, 3, 4]
```

One important rule about `insert()`: the first argument must always be a plain integer index, never a list or anything else.

```python
list1.insert([1, 2], 100)   # wrong: the index argument must be a single integer
```

```
TypeError
```

### 9. How do I add a value inside a list that is nested deep inside another list?

This builds directly on the nested indexing idea covered above. Once you have correctly navigated down to the specific inner list you want, you can call `append()` on it directly, exactly as if it were any other list.

```python
list1 = [10, 20, [300, 400, [5000, 6000], 500], 30, 40]

list1[2][2].append(7000)   # navigate to the innermost list, then append 7000 to it

print(list1)
```

```
[10, 20, [300, 400, [5000, 6000, 7000], 500], 30, 40]
```

Here, `list1[2]` gets you `[300, 400, [5000, 6000], 500]`, and `[2]` on that result gets you specifically `[5000, 6000]`, the innermost list, so calling `.append(7000)` on that exact reference correctly extends only that specific nested list, leaving everything else in the structure untouched.

### 10. How do I join a list of strings into a single string?

`join()` is used specifically to combine a list, or any iterable, made entirely of strings, into one single string, with a separator of your choice placed in between each piece.

```python
words = ['a', 'b', 'c']

s = "-".join(words)   # join every string in the list using "-" as the separator

print(s)
```

```
a-b-c
```

A common mistake is trying to use `join()` directly on a list that contains something other than plain strings, such as tuples.

```python
l = [(1, 2), (3, 4)]

", ".join(l)   # this fails, since join() only works directly on strings
```

The fix is to explicitly convert every item to a string first, right inside the same line, using a generator expression.

```python
", ".join(str(item) for item in l)   # convert each item to a string before joining
```

## Part 3: Dictionaries

### 11. Why did my dictionary silently drop one of my duplicate keys?

```python
data = {
    "Kiran": 34,
    "Sushil": 67,
    "Harsh": 34,
    "Harsh": 90
}

print(data)
```

```
{'Kiran': 34, 'Sushil': 67, 'Harsh': 90}
```

At first glance, it looks like Python simply ignored one of the entries, but that is exactly how dictionaries are designed to work. The core rule is that dictionary keys must always be unique, while the values attached to those keys are free to repeat as much as you like.

This means having several different keys sharing the same value is completely fine.

```python
{"Kiran": 34, "Harsh": 34, "Sushil": 34}   # perfectly valid, values can repeat
```

But having the exact same key appear twice is not something a dictionary can represent at all, since a dictionary is fundamentally a mapping of unique keys to values. When you write `"Harsh"` twice, the second occurrence simply overwrites the value stored under the first one, since they are the same key, not two separate entries. Understanding this single rule makes later dictionary operations, like searching, updating, and deleting entries, click into place much more easily.

## Part 4: Functions, Arguments and Lambda

### 12. How do I write a function that accepts any number of arguments, without knowing in advance how many?

Suppose you start with a simple function.

```python
def add(a, b):
    return a + b   # only ever works with exactly two arguments
```

This works fine until you try calling it with a different number of arguments.

```python
add(4, 5, 6)   # calling with three arguments instead of two
```

```
TypeError: add() takes 2 positional arguments but 3 were given
```

Writing separate versions of the function for three arguments, then four, then any other number, clearly does not scale. The actual solution is `*args`, which lets a function accept any number of positional arguments at all.

```python
def add(*nums):
    return sum(nums)   # nums arrives as a tuple containing every argument passed in
```

Now the same single function handles any number of inputs.

```python
add(4, 5)
add(4, 5, 6)
add(4, 5, 6, 7, 8)
```

Behind the scenes, when you call `add(4, 5, 6, 7)`, Python automatically packs all four values together into a tuple before the function body even runs.

```python
nums = (4, 5, 6, 7)   # this is what *args actually looks like inside the function
```

So `*args` is not magic, it is simply Python's way of saying: give me however many positional arguments you have, and I will pack them all into a tuple for you to work with inside the function.

### 13. Why do we need lambda functions when we already have def?

The first reaction most people have to lambda functions is that they seem pointless, since you can already write the exact same logic using a regular function.

```python
def square(x):
    return x * x
```

```python
square = lambda x: x * x   # does exactly the same thing as the def version above
```

Both produce the identical result, so the point of lambda is not that it is somehow better than `def`. It exists for a different, narrower purpose entirely.

Imagine you only need a small function exactly once, perhaps just to pass directly into another function. Formally defining it with `def`, giving it a permanent name, and never using that name again anywhere else, feels like unnecessary ceremony. That is precisely the situation lambda is designed for. A lambda function is simply a small, anonymous function, one without a permanent name attached to it, and it works best specifically when you only need the function once, the function is just a single expression, and you want to hand that function directly to something else, such as `map()`, `filter()`, or `sorted()`.

It is also worth knowing that a lambda function is still a completely ordinary function underneath. You can assign it to a variable exactly like `square = lambda x: x * x` above, and at that point the only real difference between it and a `def` function is simply how it was written, not how it behaves or how it gets used afterward.

## Part 5: Iterators and the Functional Tools Built On Top of Them

### 14. Why do zip(), enumerate(), map() and filter() not show me the values directly?

All four of these functions return something called an iterator in Python 3, rather than handing you back a plain list immediately.

An iterator specifically means the values are generated one at a time, on demand, rather than all being calculated and stored in memory upfront. This makes iterators memory efficient, especially useful with very large datasets, but it also means you cannot directly see the contents of an iterator just by printing it. To actually view the values, you need to convert the iterator into something concrete, most commonly a list.

```python
list(some_iterator)   # forces every value out of the iterator into an actual list
```

Keeping this single idea in mind removes a lot of confusion around the four functions covered next.

### 15. How does enumerate() work, and when do I need it?

`enumerate()` is used specifically whenever you need both the index position and the value together, at the same time, while looping over something.

Syntax:

```python
enumerate(iterable, start=0)
```

```python
fruits = ['apple', 'banana', 'mango']

for index, fruit in enumerate(fruits):
    print(index, fruit)   # unpacks both the position and the value on each loop
```

```
0 apple
1 banana
2 mango
```

You are also free to start counting from a different number instead of the default 0.

```python
for index, fruit in enumerate(fruits, start=1):   # start counting from 1 instead
    print(index, fruit)
```

```
1 apple
2 banana
3 mango
```

### 16. How does zip() combine multiple lists together?

`zip()` is used to combine several iterables together, pairing up items that share the same position across each of them.

Syntax:

```python
zip(iter1, iter2, ...)
```

```python
names = ['A', 'B', 'C']
marks = [10, 20, 30]

z = zip(names, marks)   # returns a zip object, an iterator, not a list yet

print(list(z))            # convert it to a list to actually see the paired values
```

```
[('A', 10), ('B', 20), ('C', 30)]
```

One important detail to remember: `zip()` always stops as soon as the shortest iterable given to it runs out of values, even if the other iterables still have more values left.

### 17. Why does filter() ask me for a function as an argument?

The first time you see `filter()` used, it raises two natural questions. Why does it need a function at all, and why does it return an object instead of a plain list?

```python
nums = [4, 2, 9, 7, 6, 8]

evens = filter(lambda n: n % 2 == 0, nums)   # keep only numbers where this returns True

evens = list(evens)   # convert the filter object into an actual list to see the values

print(evens)
```

```
[2, 4, 6, 8]
```

Here is what clears up the confusion. `filter()` does not check just one value, it checks every single value in the iterable you give it, effectively behaving like this internally.

```python
for n in nums:
    if condition(n):
        keep_it(n)
```

The catch is that `filter()` itself has no idea what your specific condition should be. Maybe you want even numbers, maybe you want numbers greater than 10, maybe you want strings longer than five characters. That condition changes depending on what you are trying to do, so instead of hardcoding one fixed rule, `filter()` simply asks you to hand it a function that decides, for every element, whether to keep it or not. That is exactly what `lambda n: n % 2 == 0` is doing, returning `True` to keep a value and `False` to discard it. And since `filter()` returns a filter object rather than a list directly, you generally wrap the whole call in `list()` whenever you want to actually see the filtered results.

### 18. Why does map() also ask me for a function, and how is it different from filter()?

```python
evens = [4, 2, 6, 8]

doubles = list(map(lambda n: n * 2, evens))   # transform every element by doubling it

print(doubles)
```

```
[8, 4, 12, 16]
```

The key difference from `filter()` is worth noticing directly. `map()` is not removing anything, and it is not adding any new elements either, it is transforming every single existing element into something else, one to one. Internally, it behaves something like this.

```python
result = []
for n in evens:
    result.append(transform(n))   # apply the transformation to each element
```

Just like with `filter()`, `map()` has no built-in idea of what transformation you actually want. Maybe you want to double every number, maybe you want to square them, maybe you want to convert a list of strings to uppercase, or convert temperatures from Celsius to Fahrenheit. Since that transformation logic changes every time, `map()` asks you to supply a function describing exactly how each element should be transformed, which is precisely the role `lambda n: n * 2` is playing here. And exactly like `filter()`, `map()` also returns a map object rather than a plain list, so it usually gets wrapped in `list()` too.

### 19. What is a quick reference for what these common functions actually return?

Since so many of the functions above return something other than a plain list, having this table in one place is genuinely useful to glance back at.

| Function | Return Type |
|---|---|
| `zip()` | zip object (an iterator) |
| `enumerate()` | enumerate object (an iterator) |
| `map()` | map object (an iterator) |
| `filter()` | filter object (an iterator) |
| `append()` | `None` |
| `insert()` | `None` |
| `sort()` | `None` |
| `sorted()` | a brand new sorted list |
| `join()` | a string |

The pattern worth internalizing here is that anything meant to be looped over lazily, like `zip`, `enumerate`, `map` and `filter`, returns an iterator object, while anything that directly mutates a list in place, like `append`, `insert` and `sort`, returns `None`, since their entire job is to change the original object, not to hand you back a new one.

## Part 6: Sorting and Understanding None Returns

### 20. Why did storing the result of append() in a variable give me None?

```python
nums = [1, 2, 3]

x = nums.append(5)   # this looks reasonable, but is actually a common mistake

print(x)
```

```
None
```

`append()` modifies the original list directly and in place, and its job ends there, so it does not return the updated list back to you as well. It simply returns `None`, since that is what functions in Python return by default when they do not explicitly return anything else. The correct way to use `append()` is to just call it on its own line, and then look at the original list afterward, rather than trying to capture its return value.

```python
nums.append(5)   # correct: just call append, the original list is already updated
print(nums)
```

```
[1, 2, 3, 5]
```

### 21. What is the actual difference between sort() and sorted()?

These two look similar enough to easily mix up, but they behave quite differently.

`sort()` modifies the original list directly, in place, and, matching the pattern from the previous question, it returns `None`.

```python
nums = [3, 1, 2]
nums.sort()   # sorts nums in place, does not return a new list

print(nums)
```

```
[1, 2, 3]
```

`sorted()` instead leaves the original list completely untouched, and returns a brand new sorted list for you to store separately.

```python
nums = [3, 1, 2]
new_nums = sorted(nums)   # returns a new sorted list, original nums is unchanged

print(nums)
print(new_nums)
```

```
[3, 1, 2]
[1, 2, 3]
```

A simple way to remember the difference: `sort()` sorts the list you already have, quietly, and gives you nothing back. `sorted()` leaves your original list alone and hands you a completely new one instead.

## Part 7: Bonus, Why NumPy Arrays Exist When Python Already Has Lists

This last one is not strictly a Python core language feature, but it is exactly the kind of thing that quietly confuses people once they start doing any data-heavy work in Python, so it earns a place in this note.

### 22. If Python already has lists, why do we even need NumPy arrays?

Both a Python list and a NumPy array can hold a sequence of numbers, and at a glance they look almost interchangeable.

```python
# Python List
nums = [1, 2, 3, 4]

# NumPy Array
import numpy as np
arr = np.array([1, 2, 3, 4])   # numpy needs to be imported first
```

The real difference is not in what they can store, it is entirely in how that data is actually organized in memory.

A Python list does not directly store the numbers themselves. Instead, it stores references, essentially pointers, to separate Python integer objects sitting elsewhere in memory. Conceptually, it looks something like this.

```
List
 │
 ├──► Integer Object (1)
 ├──► Integer Object (2)
 ├──► Integer Object (3)
 └──► Integer Object (4)
```

Every single integer here is its own separate Python object, complete with its own metadata overhead, and the list itself is really just a collection of addresses pointing out to each of them.

A NumPy array works completely differently. It stores its values contiguously, meaning packed directly next to each other in one continuous block of memory, with no extra Python objects and no extra pointers involved at all.

```
[1][2][3][4]
```

Because of this design, NumPy arrays end up being considerably more memory efficient, faster to read from and write to, and dramatically better suited for mathematical computation on large amounts of data. For a small list of ten or a hundred numbers, this difference is basically invisible in practice. But once you are working with millions, or even billions, of values, this single design decision around memory layout becomes the entire reason NumPy exists, and it is exactly why nearly every serious data science and machine learning library in the Python ecosystem is built directly on top of NumPy rather than on plain Python lists.

## Final Thoughts

None of these twenty two ideas are complicated on their own. What makes them worth collecting in one place is that each one is small enough to be skipped over quickly while first learning Python, yet each one is exactly the kind of thing that resurfaces later and causes a genuinely confusing bug, right at the moment you least expect it. Revisiting this list occasionally, especially the reference and mutability concepts in Part 1, since almost everything else quietly depends on them, is time well spent.
