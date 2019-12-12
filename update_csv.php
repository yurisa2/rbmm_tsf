<?php


$string = file_get_contents('php://input');

$line = json_decode($string);

// var_dump($line[0]);

$line = implode(",", $line[0]);

$file = file_get_contents('http://ti.sa2.com.br/rbmm/optimizations/optim_results.csv');


$file .= $line;
$file .= "\r\n";
// var_dump($file);

file_put_contents('optimizations/optim_results.csv', $file);
