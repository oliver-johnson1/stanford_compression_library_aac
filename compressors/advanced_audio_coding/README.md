# Advanced Audio Coding (AAC) Compression Model
This repository focuses on the Advanced Audio Coding compression model, and contributes to the [Stanford Compression Library's](https://github.com/kedartatwawadi/stanford_compression_library) Github repository. 

<!-- TABLE OF CONTENTS -->
<details>
  <summary> Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#team-members">Team Members</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
Advanced Audio Coding (AAC) is an audio coding standard for lossy audio compression that is used for storage and transmission of digital audio. It exploits the limitations of human hearing to reduce information in parts of the signal that are less perceivable. It is the successor to the MP3 format, and has especially better audio quality in the 32-80 bitrate range. It is used as the default media format by YouTube, iPhone, iPod, iPad, Apple iTunes, and several other platforms. As part of this project, we implemented a version of AAC in Python that retains many of the key attributes, but is also simple enough to understand, and we also prototyped different parameters and blocks.

## Prerequisites
For general prerequisties, please see the main directory's README of this [repository](https://github.com/oliver-johnson1/stanford_compression_library_aac).

For our specific compression model, we have added the neccessary dependencies in the main directory's requirements file. Please install them as instructed on the main directory's README if you have not already.

<!-- USAGE -->
## Usage
Just in case, make sure you run

```sh
export PYTHONPATH=$PYTHONPATH:/path/to/stanford_compression_library_aac 
```
before running any Stanford Compression Library code.

To run any of the files for AAC, please run from the main directory.

To compress a wav file into an AAC-wav file using our AAC compression model, run 
  ```sh
  python compressors/advanced_audio_coding/bitstream_formatter.py
  ```

This will also generate a spectrogram of the original wav file and the AAC-wav file to compare. It also prints the result values from the report (MSE and kbps).


To run all tests (including tests not within the AAC folder) run
```sh
find . -name "*.py" -exec py.test -s -v {} +
```

To run a single test (e.g.)
```sh
py.test -s -v compressors/advanced_audio_coding/quantization.py
```

<!-- CONTACT -->
## Team Members
* [Audrey Lee](https://github.com/Audrey-Lee88)
* [Oliver Johnson](https://github.com/oliver-johnson1)

<!-- Slides and Report -->
## Slides and Report
* [Report](AAC_Report.pdf)
* [AAC Presentation](AAC_Presentation.pdf)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgments
Thank you to our wonderful mentors and teaching team in the EE 274: Data Compression: Theory and Applications course at Stanford. 