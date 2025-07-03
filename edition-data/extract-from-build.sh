#!/bin/sh

docker_path="$1"

if [ -z "$docker_path" ]; then
  echo "Usage: $0 /path/to/dir/with/Dockerfile"
  exit 1
fi

docker build -o - --target build "$docker_path" |
  tar -x -f- --strip-components=4 home/gradle/faust-gen/build/uris.json home/gradle/faust-gen/build/www/data/genetic_bar_graph.json
