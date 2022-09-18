#!/bin/bash
# conda env create --file ai_expo_2022.yml
conda env export | grep -v "^prefix: " > ai_expo_2022.yml
