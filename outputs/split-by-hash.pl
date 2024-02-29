#!/usr/bin/perl

use strict;
use warnings;

use JSON::XS;

my ($in_file, $out_cnn, $out_xsum) = @ARGV;
die "Usage" unless(@ARGV == 3);

die "Output already exists" if(-s $out_cnn or -s $out_xsum);

open my $in_h, '<:utf8', $in_file or die "Couldn't open $in_file";
open my $cnn_h, '>:utf8', $out_cnn or die "Couldn't write CNN";
open my $xsum_h, '>:utf8', $out_xsum or die "Couldn't write XSum";

while(<$in_h>) {
  next unless(m/^\{/);
  my $struct = decode_json($_);
  my $len = length $$struct{'hash'};
  if($len == 40) {
    print $cnn_h $_;
  } elsif($len == 8) {
    print $xsum_h $_;
  } else {
    die "Weird length: $len";
  }
}

