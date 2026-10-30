# theremin.util

Utils for theremin.

### Functions

| [`annotate_with`](#theremin.util.annotate_with)(annotation_type)               | Decorator to annotate a function with a specified type and store the annotation in the correct scope's `__annotations__` dictionary.   |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| [`current_time_string_with_milliseconds`](#theremin.util.current_time_string_with_milliseconds)()      | Get the current time with milliseconds, as a string.                                                                                   |
| [`ensure_plain_types`](#theremin.util.ensure_plain_types)(obj)                      | Recursively convert numpy numbers/arrays in common containers to builtin types.                                                        |
| [`format_dict_values`](#theremin.util.format_dict_values)(input_dict[, max_digits]) | Formats all numerical values in a dictionary to strings with a specified number of significant digits.                                 |
| `format_float`(value[, ndigits])                                                              |                                                                                                                                        |
| [`format_label_xyz`](#theremin.util.format_label_xyz)(label, x, y, z, \*[, ...])  | Format the label and coordinates with customizable widths.                                                                             |
| [`format_milliseconds_time`](#theremin.util.format_milliseconds_time)(timestamp)          | Format milliseconds as a string.                                                                                                       |
| [`json_lines`](#theremin.util.json_lines)(string_or_path_to_string)         | Parse a file or string containing JSON-like dictionaries on each line.                                                                 |
| [`obfuscate_args`](#theremin.util.obfuscate_args)(func, keep_args)              | Creates a new function with only the specified arguments, with other arguments fixed to their defaults.                                |
| [`return_none`](#theremin.util.return_none)(\*args, \*\*kwargs)              | An empty function that returns None no matter the arguments.                                                                           |
| [`to_builtin_number`](#theremin.util.to_builtin_number)(x)                         | Convert numpy numeric scalars to builtin numbers; leave others unchanged.                                                              |

### Classes

| [`AllZerosDict`](#theremin.util.AllZerosDict)(\*args, \*\*kwargs)   | A dict that only returns 0.0 for all keys.   |
|-------------------------------------------------------------------------------------|----------------------------------------------|
| `HandLandmarkIndex`()                                                               |                                              |

### *class* theremin.util.AllZerosDict(\*args, \*\*kwargs)

Bases: [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

A dict that only returns 0.0 for all keys.

```pycon
>>> all_zeros_dict = AllZerosDict()
>>> all_zeros_dict['anything']
0.0
```

### theremin.util.annotate_with(annotation_type)

Decorator to annotate a function with a specified type and store the annotation
in the correct scope’s `__annotations__` dictionary.

* **Parameters:**
  **annotation_type** (*Any*) – The type to annotate the function with.
* **Returns:**
  The original function with the annotation added.
* **Return type:**
  function

### Examples

```pycon
>>> @annotate_with('int')
... def global_func():
...     pass
...
>>> global_func()
>>> __annotations__['global_func']
'int'
```

### theremin.util.current_time_string_with_milliseconds()

Get the current time with milliseconds, as a string.

### theremin.util.ensure_plain_types(obj)

Recursively convert numpy numbers/arrays in common containers to builtin types.

Supports dicts, lists, tuples, sets; leaves other types unchanged.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### theremin.util.format_dict_values(input_dict, max_digits=8)

Formats all numerical values in a dictionary to strings with a specified number of significant digits.

* **Parameters:**
  * **input_dict** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The dictionary whose numerical values will be formatted.
  * **max_digits** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – The maximum number of significant digits to use in the formatting.
* **Returns:**
  A new dictionary with the same keys as input_dict, but with numerical values replaced by formatted strings.
* **Return type:**
  [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Example

```pycon
>>> original_dict = {'a': 123456789, 'b': 1234.56789, 'c': 12.345789, 'd': 0.0012345789}
>>> format_dict_values(original_dict, 5)
{'a': '1.2346e+08', 'b': '1234.6', 'c': '12.346', 'd': '0.0012346'}
```

#### NOTE
Non-numeric values are left unchanged.

### theremin.util.format_label_xyz(label, x, y, z, , label_width=15, coord_width=8)

Format the label and coordinates with customizable widths.

* **Parameters:**
  * **label** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The label for the coordinates (e.g., “Right Wrist:”).
  * **x** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The coordinates to format.
  * **y** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The coordinates to format.
  * **z** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The coordinates to format.
  * **label_width** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – The width of the label field.
  * **coord_width** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – The width of the coordinate fields.
* **Returns:**
  The formatted string.
* **Return type:**
  [*str*](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> format_label_xyz('Right Wrist:', 1.2, 3.4, 5.6)
'Right Wrist:    x=     1.2 y=     3.4 z=     5.6'
>>> format_label_xyz('Right Wrist:', 1.2, 3.4, 5.6, label_width=10, coord_width=4)
'Right Wrist: x= 1.2 y= 3.4 z= 5.6'
```

### theremin.util.format_milliseconds_time(timestamp)

Format milliseconds as a string.

### theremin.util.json_lines(string_or_path_to_string)

Parse a file or string containing JSON-like dictionaries on each line.
Yields only the lines that can be deserialized to dictionaries.

* **Parameters:**
  **string_or_path_to_string** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – A file path or the content as a string
* **Yields:**
  *dict* – Each successfully parsed dictionary
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### theremin.util.obfuscate_args(func, keep_args)

Creates a new function with only the specified arguments,
with other arguments fixed to their defaults.

### theremin.util.return_none(\*args, \*\*kwargs)

An empty function that returns None no matter the arguments.
Often used as a “do nothing” general callback function.

### theremin.util.to_builtin_number(x)

Convert numpy numeric scalars to builtin numbers; leave others unchanged.

- numpy floating -> float
- numpy integer -> int
- python numbers.Number -> unchanged
- numpy arrays -> x.tolist() if 0-d array then recurse to scalar
- other types -> unchanged

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
