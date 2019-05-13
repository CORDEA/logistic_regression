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

use strict;
use warnings;

use Math::MatrixReal;

my $data = Math::MatrixReal->new_from_rows(
    [
        [0, 0, 0, 0]
    ]
);

my $answer = Math::MatrixReal->new_from_rows(
    [
        [0]
    ]
);

my $number_of_learnings = 1000;
my $alpha = 0.01;

sub _sigmoid {
    my $x = shift;
    return 1 / (1 + exp(-$x));
}

sub _logistic {
    my $i = 0;
    my ($row, $column) = $data->dim();
    my $theta = Math::MatrixReal->new($column, 1);
    my $score;
    while ($i < $number_of_learnings) {
        $score = $data * $theta;
        $score = $score->each(sub { _sigmoid(shift) });
        $theta -= $alpha * ~$data * ($score - $answer) / $row;
        ++$i;
    }
    return $theta;
};

my $theta = _logistic();
print $theta;
