#!/usr/bin/env bash
# obtains all data tables from database
mkdir groundtruth
TS=`sqlite3 groundtruth.sqlite "SELECT tbl_name FROM sqlite_master  WHERE type='table' and tbl_name not like 'metadata%' and tbl_name not like 'sqlite_%' and tbl_name!='dataset_profile';"`
for T in $TS; do
sqlite3 groundtruth.sqlite <<!
.headers on
.mode csv
.output groundtruth/$T.csv
select * from "$T";
!

done