#!/bin/bash

for file in ./*.jpg; do
  convert $file -resize 28x28\! $file
done
