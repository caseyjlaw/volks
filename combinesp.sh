#!/bin/bash

rm -f $1.sp

ls -1 *.sp | xargs cat >> $1.sp
