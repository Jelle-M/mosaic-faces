#!/usr/bin/env bash
rm -rf ../faces/*
rm -rf ../out/*
rm -rf ../tiles/*

if [ ! -z $1 ] ; then
    rm -rf ../frames/*
    rm ../extracted.txt
fi
