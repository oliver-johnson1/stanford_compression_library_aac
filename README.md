# Stanford Compression Library (AAC Edition!)
This fork of the [Stanford Compression Library](https://github.com/kedartatwawadi/stanford_compression_library) includes a compression algorithm on Advanced Audio Coding (AAC).

For a more detailed explanation on AAC, please see the README in the [AAC](compressors/advanced_audio_coding) folder.

## Compression algorithms
Here is a list of algorithms used for AAC:
- [Huffman codes](compressors/huffman_coder.py)
- [Advanced Audio Coding](compressors/advanced_audio_coding)


## Getting started
- Create conda environment and install required packages:
    ```
    conda create --name myenv python=3.8.2
    conda activate myenv
    python -m pip install -r requirements.txt
    ```
- Add path to the repo to `PYTHONPATH`:
    ```
    export PYTHONPATH=$PYTHONPATH:<path_to_repo>
    ``` 

- **Run unit tests**

  To run all tests:
    ```
    find . -name "*.py" -exec py.test -s -v {} +
    ```

  To run a single test
  ```
  py.test -s -v core/data_stream_tests.py
  ```

