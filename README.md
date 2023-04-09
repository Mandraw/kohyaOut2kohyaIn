# kohyaOut2kohyaIn

So I wanted to easily take a lora's training parameters and use them in kohya...
So I made this thing ^^"

This Python script converts an input JSON file in a specific format to an output JSON file in a different format, with the values mapped to different keys. Any key that doesn't have a corresponding value in the input file is set to `null` in the output file.

## Requirements

This script requires Python 3 to run. No external libraries or dependencies are needed.

## Usage

To use the script, run the following command in a terminal or command prompt:

```bash
python kohyaOut2kohyaIn.py -n <input_file> -o <output_file>
```
OR
```bash
python kohyaOut2kohyaIn.py --input <input_file> --output <output_file>
```

Replace `<input_file>` with the name of the input JSON file, and `<output_file>` with the desired name for the output JSON file.

## Notes
The script assumes that the input file is in the format specified above. If the input file is not in this format, the script may not work correctly.
The script does not modify the input file. It only creates a new output file with the values mapped to different keys.