#!/bin/sh

# get all files starting with input_* from testes-takuzu/
# and run them, comparing them with their respective output_*
# if the output is the same, print "Test _ SUCCESS" in green,
# otherwise print "Test _ FAILED" in red

files=$(ls testes-takuzu/ | grep "^input_")

for file in $files
do
  python takuzu.py < testes-takuzu/$file > /tmp/takuzu.out
  output=$(cat /tmp/takuzu.out)
  output_file=$(echo $file | sed 's/input/output/')
  expected_output=$(cat testes-takuzu/$output_file)
  if [ "$output" = "$expected_output" ]
  then
    echo -e "\e[32mTest $file SUCCESS\e[0m"
  else
    echo -e "\e[31mTest $file FAILED\e[0m"
    # diff between /tmp/takuzu.out and testes-takuzu/$output_file
    colordiff -u testes-takuzu/$output_file /tmp/takuzu.out
  fi
done