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

package LogisticRegression;

use strict;
use warnings;

use Math::MatrixReal;

sub new {
    my $class = shift;
    my ($data, $answer, $number_of_learnings, $alpha) = @_;
    bless {
        data => $data,
        answer => $answer,
        number_of_learnings => $number_of_learnings,
        alpha => $alpha
    }, $class
}

sub _sigmoid {
    my $x = shift;
    return 1 / (1 + exp(-$x));
}

sub logistic {
    my $self = shift;
    my $data = $self->{data};
    my $alpha = $self->{alpha};
    my $answer = $self->{answer};
    my $number_of_learnings = $self->{number_of_learnings};

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
}

sub predict {
    my ($self, $theta) = @_;
    my $score = $self->{data} * $theta;
    $score = $score->each(sub { _sigmoid(shift) });
    return $score->each(sub { shift >= 0.5 });
}

1;
