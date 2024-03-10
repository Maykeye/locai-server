#!/bin/bash

DATA=$(cat << END
{
	"code":"def hello world:\nprint(\"Hello\")",
	"line":1, 
	"col":4
}
END
)

curl  http://127.0.0.1:13333/gen -d "$DATA"
