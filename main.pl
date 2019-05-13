# Copyright 2019 Yoshihiro Tanaka
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Author: Yoshihiro Tanaka <contact@cordea.jp>
# date  : 2019-05-13

use lib "./lib";

use strict;
use warnings;

use Text::CSV_XS;
use Math::MatrixReal;
use LogisticRegression;

my $csv_file_name = $ARGV[0];
my $answer_file_name = $ARGV[1];

my $csv = Text::CSV_XS->new({binary => 1, auto_diag => 1});
my @lines = ();
open my $fh, "<:encoding(utf8)", $csv_file_name or die "$csv_file_name: $!";
while (my $row = $csv->getline($fh)) {
    push @lines, \@$row;
}
close $fh;

open IN, $answer_file_name;
my @raw_answer = <IN>;
close IN;
@raw_answer = map { [$_ == 0 ? 0 : 1] } @raw_answer;

my $data = Math::MatrixReal->new_from_rows(\@lines);
my $answer = Math::MatrixReal->new_from_rows(\@raw_answer);

my $regression = LogisticRegression->new($data, $answer, 1000, 0.01);
my $theta = $regression->logistic();
print $answer == $regression->predict($theta);
