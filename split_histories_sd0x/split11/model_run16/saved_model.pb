��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02v2.3.0-rc2-23-gb36436b0878��
�
conv1d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv1d_32/kernel
y
$conv1d_32/kernel/Read/ReadVariableOpReadVariableOpconv1d_32/kernel*"
_output_shapes
: *
dtype0
t
conv1d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_32/bias
m
"conv1d_32/bias/Read/ReadVariableOpReadVariableOpconv1d_32/bias*
_output_shapes
: *
dtype0
�
conv1d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv1d_33/kernel
y
$conv1d_33/kernel/Read/ReadVariableOpReadVariableOpconv1d_33/kernel*"
_output_shapes
:  *
dtype0
t
conv1d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_33/bias
m
"conv1d_33/bias/Read/ReadVariableOpReadVariableOpconv1d_33/bias*
_output_shapes
: *
dtype0
{
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d* 
shared_namedense_32/kernel
t
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes
:	�d*
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:d*
dtype0
z
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_33/kernel
s
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes

:d*
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/conv1d_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv1d_32/kernel/m
�
+Adam/conv1d_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_32/kernel/m*"
_output_shapes
: *
dtype0
�
Adam/conv1d_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv1d_32/bias/m
{
)Adam/conv1d_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_32/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv1d_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv1d_33/kernel/m
�
+Adam/conv1d_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_33/kernel/m*"
_output_shapes
:  *
dtype0
�
Adam/conv1d_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv1d_33/bias/m
{
)Adam/conv1d_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_33/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*'
shared_nameAdam/dense_32/kernel/m
�
*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m*
_output_shapes
:	�d*
dtype0
�
Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_32/bias/m
y
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes
:d*
dtype0
�
Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_33/kernel/m
�
*Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/m*
_output_shapes

:d*
dtype0
�
Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/m
y
(Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv1d_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv1d_32/kernel/v
�
+Adam/conv1d_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_32/kernel/v*"
_output_shapes
: *
dtype0
�
Adam/conv1d_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv1d_32/bias/v
{
)Adam/conv1d_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_32/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv1d_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv1d_33/kernel/v
�
+Adam/conv1d_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_33/kernel/v*"
_output_shapes
:  *
dtype0
�
Adam/conv1d_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv1d_33/bias/v
{
)Adam/conv1d_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_33/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*'
shared_nameAdam/dense_32/kernel/v
�
*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v*
_output_shapes
:	�d*
dtype0
�
Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_32/bias/v
y
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes
:d*
dtype0
�
Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_33/kernel/v
�
*Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/v*
_output_shapes

:d*
dtype0
�
Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/v
y
(Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�4
value�4B�4 B�4
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		keras_api

regularization_losses

signatures
	variables
trainable_variables
h

kernel
bias
	keras_api
regularization_losses
	variables
trainable_variables
h

kernel
bias
	keras_api
regularization_losses
	variables
trainable_variables
R
	keras_api
regularization_losses
	variables
trainable_variables
R
	keras_api
regularization_losses
 	variables
!trainable_variables
R
"	keras_api
#regularization_losses
$	variables
%trainable_variables
h

&kernel
'bias
(	keras_api
)regularization_losses
*	variables
+trainable_variables
h

,kernel
-bias
.	keras_api
/regularization_losses
0	variables
1trainable_variables
�
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy
�

7layers
8non_trainable_variables
trainable_variables
9layer_metrics
	variables

regularization_losses
:layer_regularization_losses
;metrics
 
 
8
0
1
2
3
&4
'5
,6
-7
8
0
1
2
3
&4
'5
,6
-7
\Z
VARIABLE_VALUEconv1d_32/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_32/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
�
<non_trainable_variables
trainable_variables
=layer_metrics
regularization_losses
	variables

>layers
?layer_regularization_losses
@metrics
 

0
1

0
1
\Z
VARIABLE_VALUEconv1d_33/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_33/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
�
Anon_trainable_variables
trainable_variables
Blayer_metrics
regularization_losses
	variables

Clayers
Dlayer_regularization_losses
Emetrics
 

0
1

0
1
�
Fnon_trainable_variables
trainable_variables
Glayer_metrics
regularization_losses
	variables

Hlayers
Ilayer_regularization_losses
Jmetrics
 
 
 
�
Knon_trainable_variables
!trainable_variables
Llayer_metrics
regularization_losses
 	variables

Mlayers
Nlayer_regularization_losses
Ometrics
 
 
 
�
Pnon_trainable_variables
%trainable_variables
Qlayer_metrics
#regularization_losses
$	variables

Rlayers
Slayer_regularization_losses
Tmetrics
 
 
 
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
�
Unon_trainable_variables
+trainable_variables
Vlayer_metrics
)regularization_losses
*	variables

Wlayers
Xlayer_regularization_losses
Ymetrics
 

&0
'1

&0
'1
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
�
Znon_trainable_variables
1trainable_variables
[layer_metrics
/regularization_losses
0	variables

\layers
]layer_regularization_losses
^metrics
 

,0
-1

,0
-1
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6
 
 
 

_0
`1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	atotal
	bcount
c	keras_api
d	variables
D
	etotal
	fcount
g
_fn_kwargs
h	keras_api
i	variables
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

d	variables

a0
b1
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

i	variables

e0
f1
}
VARIABLE_VALUEAdam/conv1d_32/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_32/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_33/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_33/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_33/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_33/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_32/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_32/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_33/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_33/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_33/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_33/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_conv1d_32_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_32_inputconv1d_32/kernelconv1d_32/biasconv1d_33/kernelconv1d_33/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_800466
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_32/kernel/Read/ReadVariableOp"conv1d_32/bias/Read/ReadVariableOp$conv1d_33/kernel/Read/ReadVariableOp"conv1d_33/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv1d_32/kernel/m/Read/ReadVariableOp)Adam/conv1d_32/bias/m/Read/ReadVariableOp+Adam/conv1d_33/kernel/m/Read/ReadVariableOp)Adam/conv1d_33/bias/m/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp*Adam/dense_33/kernel/m/Read/ReadVariableOp(Adam/dense_33/bias/m/Read/ReadVariableOp+Adam/conv1d_32/kernel/v/Read/ReadVariableOp)Adam/conv1d_32/bias/v/Read/ReadVariableOp+Adam/conv1d_33/kernel/v/Read/ReadVariableOp)Adam/conv1d_33/bias/v/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp*Adam/dense_33/kernel/v/Read/ReadVariableOp(Adam/dense_33/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_800863
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_32/kernelconv1d_32/biasconv1d_33/kernelconv1d_33/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv1d_32/kernel/mAdam/conv1d_32/bias/mAdam/conv1d_33/kernel/mAdam/conv1d_33/bias/mAdam/dense_32/kernel/mAdam/dense_32/bias/mAdam/dense_33/kernel/mAdam/dense_33/bias/mAdam/conv1d_32/kernel/vAdam/conv1d_32/bias/vAdam/conv1d_33/kernel/vAdam/conv1d_33/bias/vAdam/dense_32/kernel/vAdam/dense_32/bias/vAdam/dense_33/kernel/vAdam/dense_33/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_800972ր
�
d
+__inference_dropout_16_layer_call_fn_800668

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_8002262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
I__inference_sequential_16_layer_call_and_return_conditional_losses_800416

inputs
conv1d_32_800392
conv1d_32_800394
conv1d_33_800397
conv1d_33_800399
dense_32_800405
dense_32_800407
dense_33_800410
dense_33_800412
identity��!conv1d_32/StatefulPartitionedCall�!conv1d_33/StatefulPartitionedCall� dense_32/StatefulPartitionedCall� dense_33/StatefulPartitionedCall�
!conv1d_32/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_32_800392conv1d_32_800394*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_32_layer_call_and_return_conditional_losses_8001632#
!conv1d_32/StatefulPartitionedCall�
!conv1d_33/StatefulPartitionedCallStatefulPartitionedCall*conv1d_32/StatefulPartitionedCall:output:0conv1d_33_800397conv1d_33_800399*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_33_layer_call_and_return_conditional_losses_8001982#
!conv1d_33/StatefulPartitionedCall�
dropout_16/PartitionedCallPartitionedCall*conv1d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_8002362
dropout_16/PartitionedCall�
 max_pooling1d_16/PartitionedCallPartitionedCall#dropout_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_8001372"
 max_pooling1d_16/PartitionedCall�
flatten_16/PartitionedCallPartitionedCall)max_pooling1d_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_8002512
flatten_16/PartitionedCall�
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_800405dense_32_800407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_8002702"
 dense_32/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_800410dense_33_800412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_8002942"
 dense_33/StatefulPartitionedCall�
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0"^conv1d_32/StatefulPartitionedCall"^conv1d_33/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_32/StatefulPartitionedCall!conv1d_32/StatefulPartitionedCall2F
!conv1d_33/StatefulPartitionedCall!conv1d_33/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling1d_16_layer_call_fn_800143

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_8001372
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv1d_33_layer_call_and_return_conditional_losses_800663

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :::S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_dense_33_layer_call_and_return_conditional_losses_800294

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�7
�
I__inference_sequential_16_layer_call_and_return_conditional_losses_800613

inputs9
5conv1d_32_conv1d_expanddims_1_readvariableop_resource-
)conv1d_32_biasadd_readvariableop_resource9
5conv1d_33_conv1d_expanddims_1_readvariableop_resource-
)conv1d_33_biasadd_readvariableop_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource
identity��
conv1d_32/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_32/conv1d/ExpandDims/dim�
conv1d_32/conv1d/ExpandDims
ExpandDimsinputs(conv1d_32/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d_32/conv1d/ExpandDims�
,conv1d_32/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_32_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02.
,conv1d_32/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_32/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_32/conv1d/ExpandDims_1/dim�
conv1d_32/conv1d/ExpandDims_1
ExpandDims4conv1d_32/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_32/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_32/conv1d/ExpandDims_1�
conv1d_32/conv1dConv2D$conv1d_32/conv1d/ExpandDims:output:0&conv1d_32/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_32/conv1d�
conv1d_32/conv1d/SqueezeSqueezeconv1d_32/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_32/conv1d/Squeeze�
 conv1d_32/BiasAdd/ReadVariableOpReadVariableOp)conv1d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_32/BiasAdd/ReadVariableOp�
conv1d_32/BiasAddBiasAdd!conv1d_32/conv1d/Squeeze:output:0(conv1d_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_32/BiasAddz
conv1d_32/ReluReluconv1d_32/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_32/Relu�
conv1d_33/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_33/conv1d/ExpandDims/dim�
conv1d_33/conv1d/ExpandDims
ExpandDimsconv1d_32/Relu:activations:0(conv1d_33/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d_33/conv1d/ExpandDims�
,conv1d_33/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_33_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02.
,conv1d_33/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_33/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_33/conv1d/ExpandDims_1/dim�
conv1d_33/conv1d/ExpandDims_1
ExpandDims4conv1d_33/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_33/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_33/conv1d/ExpandDims_1�
conv1d_33/conv1dConv2D$conv1d_33/conv1d/ExpandDims:output:0&conv1d_33/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_33/conv1d�
conv1d_33/conv1d/SqueezeSqueezeconv1d_33/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_33/conv1d/Squeeze�
 conv1d_33/BiasAdd/ReadVariableOpReadVariableOp)conv1d_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_33/BiasAdd/ReadVariableOp�
conv1d_33/BiasAddBiasAdd!conv1d_33/conv1d/Squeeze:output:0(conv1d_33/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_33/BiasAddz
conv1d_33/ReluReluconv1d_33/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_33/Relu�
dropout_16/IdentityIdentityconv1d_33/Relu:activations:0*
T0*+
_output_shapes
:��������� 2
dropout_16/Identity�
max_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_16/ExpandDims/dim�
max_pooling1d_16/ExpandDims
ExpandDimsdropout_16/Identity:output:0(max_pooling1d_16/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
max_pooling1d_16/ExpandDims�
max_pooling1d_16/MaxPoolMaxPool$max_pooling1d_16/ExpandDims:output:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling1d_16/MaxPool�
max_pooling1d_16/SqueezeSqueeze!max_pooling1d_16/MaxPool:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims
2
max_pooling1d_16/Squeezeu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_16/Const�
flatten_16/ReshapeReshape!max_pooling1d_16/Squeeze:output:0flatten_16/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_16/Reshape�
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02 
dense_32/MatMul/ReadVariableOp�
dense_32/MatMulMatMulflatten_16/Reshape:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_32/MatMul�
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_32/BiasAdd/ReadVariableOp�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_32/BiasAdds
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
dense_32/Relu�
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_33/MatMul/ReadVariableOp�
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_33/MatMul�
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_33/BiasAdd/ReadVariableOp�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_33/BiasAdd|
dense_33/SoftmaxSoftmaxdense_33/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_33/Softmaxn
IdentityIdentitydense_33/Softmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������:::::::::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
G
+__inference_flatten_16_layer_call_fn_800695

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_8002512
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
b
F__inference_flatten_16_layer_call_and_return_conditional_losses_800251

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_dense_32_layer_call_and_return_conditional_losses_800270

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_sequential_16_layer_call_fn_800508

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_8004162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_conv1d_32_layer_call_and_return_conditional_losses_800629

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_800137

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�

*__inference_conv1d_33_layer_call_fn_800647

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_33_layer_call_and_return_conditional_losses_8001982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

*__inference_conv1d_32_layer_call_fn_800638

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_32_layer_call_and_return_conditional_losses_8001632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_sequential_16_layer_call_fn_800487

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_8003682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_dense_33_layer_call_and_return_conditional_losses_800732

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_800466
conv1d_32_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_32_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_8001282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_32_input
�
�
E__inference_conv1d_32_layer_call_and_return_conditional_losses_800163

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
I__inference_sequential_16_layer_call_and_return_conditional_losses_800368

inputs
conv1d_32_800344
conv1d_32_800346
conv1d_33_800349
conv1d_33_800351
dense_32_800357
dense_32_800359
dense_33_800362
dense_33_800364
identity��!conv1d_32/StatefulPartitionedCall�!conv1d_33/StatefulPartitionedCall� dense_32/StatefulPartitionedCall� dense_33/StatefulPartitionedCall�"dropout_16/StatefulPartitionedCall�
!conv1d_32/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_32_800344conv1d_32_800346*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_32_layer_call_and_return_conditional_losses_8001632#
!conv1d_32/StatefulPartitionedCall�
!conv1d_33/StatefulPartitionedCallStatefulPartitionedCall*conv1d_32/StatefulPartitionedCall:output:0conv1d_33_800349conv1d_33_800351*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_33_layer_call_and_return_conditional_losses_8001982#
!conv1d_33/StatefulPartitionedCall�
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall*conv1d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_8002262$
"dropout_16/StatefulPartitionedCall�
 max_pooling1d_16/PartitionedCallPartitionedCall+dropout_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_8001372"
 max_pooling1d_16/PartitionedCall�
flatten_16/PartitionedCallPartitionedCall)max_pooling1d_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_8002512
flatten_16/PartitionedCall�
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_800357dense_32_800359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_8002702"
 dense_32/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_800362dense_33_800364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_8002942"
 dense_33/StatefulPartitionedCall�
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0"^conv1d_32/StatefulPartitionedCall"^conv1d_33/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_32/StatefulPartitionedCall!conv1d_32/StatefulPartitionedCall2F
!conv1d_33/StatefulPartitionedCall!conv1d_33/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
G
+__inference_dropout_16_layer_call_fn_800673

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_8002362
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
~
)__inference_dense_32_layer_call_fn_800710

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_8002702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
~
)__inference_dense_33_layer_call_fn_800741

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_8002942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
b
F__inference_flatten_16_layer_call_and_return_conditional_losses_800701

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_dense_32_layer_call_and_return_conditional_losses_800721

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�E
�
!__inference__wrapped_model_800128
conv1d_32_inputG
Csequential_16_conv1d_32_conv1d_expanddims_1_readvariableop_resource;
7sequential_16_conv1d_32_biasadd_readvariableop_resourceG
Csequential_16_conv1d_33_conv1d_expanddims_1_readvariableop_resource;
7sequential_16_conv1d_33_biasadd_readvariableop_resource9
5sequential_16_dense_32_matmul_readvariableop_resource:
6sequential_16_dense_32_biasadd_readvariableop_resource9
5sequential_16_dense_33_matmul_readvariableop_resource:
6sequential_16_dense_33_biasadd_readvariableop_resource
identity��
-sequential_16/conv1d_32/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_16/conv1d_32/conv1d/ExpandDims/dim�
)sequential_16/conv1d_32/conv1d/ExpandDims
ExpandDimsconv1d_32_input6sequential_16/conv1d_32/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2+
)sequential_16/conv1d_32/conv1d/ExpandDims�
:sequential_16/conv1d_32/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_16_conv1d_32_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02<
:sequential_16/conv1d_32/conv1d/ExpandDims_1/ReadVariableOp�
/sequential_16/conv1d_32/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_16/conv1d_32/conv1d/ExpandDims_1/dim�
+sequential_16/conv1d_32/conv1d/ExpandDims_1
ExpandDimsBsequential_16/conv1d_32/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_16/conv1d_32/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2-
+sequential_16/conv1d_32/conv1d/ExpandDims_1�
sequential_16/conv1d_32/conv1dConv2D2sequential_16/conv1d_32/conv1d/ExpandDims:output:04sequential_16/conv1d_32/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2 
sequential_16/conv1d_32/conv1d�
&sequential_16/conv1d_32/conv1d/SqueezeSqueeze'sequential_16/conv1d_32/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2(
&sequential_16/conv1d_32/conv1d/Squeeze�
.sequential_16/conv1d_32/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv1d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv1d_32/BiasAdd/ReadVariableOp�
sequential_16/conv1d_32/BiasAddBiasAdd/sequential_16/conv1d_32/conv1d/Squeeze:output:06sequential_16/conv1d_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2!
sequential_16/conv1d_32/BiasAdd�
sequential_16/conv1d_32/ReluRelu(sequential_16/conv1d_32/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
sequential_16/conv1d_32/Relu�
-sequential_16/conv1d_33/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_16/conv1d_33/conv1d/ExpandDims/dim�
)sequential_16/conv1d_33/conv1d/ExpandDims
ExpandDims*sequential_16/conv1d_32/Relu:activations:06sequential_16/conv1d_33/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2+
)sequential_16/conv1d_33/conv1d/ExpandDims�
:sequential_16/conv1d_33/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_16_conv1d_33_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02<
:sequential_16/conv1d_33/conv1d/ExpandDims_1/ReadVariableOp�
/sequential_16/conv1d_33/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_16/conv1d_33/conv1d/ExpandDims_1/dim�
+sequential_16/conv1d_33/conv1d/ExpandDims_1
ExpandDimsBsequential_16/conv1d_33/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_16/conv1d_33/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2-
+sequential_16/conv1d_33/conv1d/ExpandDims_1�
sequential_16/conv1d_33/conv1dConv2D2sequential_16/conv1d_33/conv1d/ExpandDims:output:04sequential_16/conv1d_33/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2 
sequential_16/conv1d_33/conv1d�
&sequential_16/conv1d_33/conv1d/SqueezeSqueeze'sequential_16/conv1d_33/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2(
&sequential_16/conv1d_33/conv1d/Squeeze�
.sequential_16/conv1d_33/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv1d_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv1d_33/BiasAdd/ReadVariableOp�
sequential_16/conv1d_33/BiasAddBiasAdd/sequential_16/conv1d_33/conv1d/Squeeze:output:06sequential_16/conv1d_33/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2!
sequential_16/conv1d_33/BiasAdd�
sequential_16/conv1d_33/ReluRelu(sequential_16/conv1d_33/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
sequential_16/conv1d_33/Relu�
!sequential_16/dropout_16/IdentityIdentity*sequential_16/conv1d_33/Relu:activations:0*
T0*+
_output_shapes
:��������� 2#
!sequential_16/dropout_16/Identity�
-sequential_16/max_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_16/max_pooling1d_16/ExpandDims/dim�
)sequential_16/max_pooling1d_16/ExpandDims
ExpandDims*sequential_16/dropout_16/Identity:output:06sequential_16/max_pooling1d_16/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2+
)sequential_16/max_pooling1d_16/ExpandDims�
&sequential_16/max_pooling1d_16/MaxPoolMaxPool2sequential_16/max_pooling1d_16/ExpandDims:output:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling1d_16/MaxPool�
&sequential_16/max_pooling1d_16/SqueezeSqueeze/sequential_16/max_pooling1d_16/MaxPool:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims
2(
&sequential_16/max_pooling1d_16/Squeeze�
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2 
sequential_16/flatten_16/Const�
 sequential_16/flatten_16/ReshapeReshape/sequential_16/max_pooling1d_16/Squeeze:output:0'sequential_16/flatten_16/Const:output:0*
T0*(
_output_shapes
:����������2"
 sequential_16/flatten_16/Reshape�
,sequential_16/dense_32/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_32_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02.
,sequential_16/dense_32/MatMul/ReadVariableOp�
sequential_16/dense_32/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential_16/dense_32/MatMul�
-sequential_16/dense_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_32_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_16/dense_32/BiasAdd/ReadVariableOp�
sequential_16/dense_32/BiasAddBiasAdd'sequential_16/dense_32/MatMul:product:05sequential_16/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2 
sequential_16/dense_32/BiasAdd�
sequential_16/dense_32/ReluRelu'sequential_16/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
sequential_16/dense_32/Relu�
,sequential_16/dense_33/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_33_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_16/dense_33/MatMul/ReadVariableOp�
sequential_16/dense_33/MatMulMatMul)sequential_16/dense_32/Relu:activations:04sequential_16/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_16/dense_33/MatMul�
-sequential_16/dense_33/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_16/dense_33/BiasAdd/ReadVariableOp�
sequential_16/dense_33/BiasAddBiasAdd'sequential_16/dense_33/MatMul:product:05sequential_16/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_16/dense_33/BiasAdd�
sequential_16/dense_33/SoftmaxSoftmax'sequential_16/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:���������2 
sequential_16/dense_33/Softmax|
IdentityIdentity(sequential_16/dense_33/Softmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������:::::::::\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_32_input
�
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_800226

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:��������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:��������� 2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:��������� 2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
.__inference_sequential_16_layer_call_fn_800387
conv1d_32_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_32_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_8003682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_32_input
�
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_800690

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:��������� 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_800972
file_prefix%
!assignvariableop_conv1d_32_kernel%
!assignvariableop_1_conv1d_32_bias'
#assignvariableop_2_conv1d_33_kernel%
!assignvariableop_3_conv1d_33_bias&
"assignvariableop_4_dense_32_kernel$
 assignvariableop_5_dense_32_bias&
"assignvariableop_6_dense_33_kernel$
 assignvariableop_7_dense_33_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1/
+assignvariableop_17_adam_conv1d_32_kernel_m-
)assignvariableop_18_adam_conv1d_32_bias_m/
+assignvariableop_19_adam_conv1d_33_kernel_m-
)assignvariableop_20_adam_conv1d_33_bias_m.
*assignvariableop_21_adam_dense_32_kernel_m,
(assignvariableop_22_adam_dense_32_bias_m.
*assignvariableop_23_adam_dense_33_kernel_m,
(assignvariableop_24_adam_dense_33_bias_m/
+assignvariableop_25_adam_conv1d_32_kernel_v-
)assignvariableop_26_adam_conv1d_32_bias_v/
+assignvariableop_27_adam_conv1d_33_kernel_v-
)assignvariableop_28_adam_conv1d_33_bias_v.
*assignvariableop_29_adam_dense_32_kernel_v,
(assignvariableop_30_adam_dense_32_bias_v.
*assignvariableop_31_adam_dense_33_kernel_v,
(assignvariableop_32_adam_dense_33_bias_v
identity_34��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*�
value�B�"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_32_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_32_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_33_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_33_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_32_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_32_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_33_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_33_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv1d_32_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv1d_32_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv1d_33_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv1d_33_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_32_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_32_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_33_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_33_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv1d_32_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv1d_32_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv1d_33_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv1d_33_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_32_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_32_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_33_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_33_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33�
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�A
�
I__inference_sequential_16_layer_call_and_return_conditional_losses_800564

inputs9
5conv1d_32_conv1d_expanddims_1_readvariableop_resource-
)conv1d_32_biasadd_readvariableop_resource9
5conv1d_33_conv1d_expanddims_1_readvariableop_resource-
)conv1d_33_biasadd_readvariableop_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource
identity��
conv1d_32/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_32/conv1d/ExpandDims/dim�
conv1d_32/conv1d/ExpandDims
ExpandDimsinputs(conv1d_32/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d_32/conv1d/ExpandDims�
,conv1d_32/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_32_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02.
,conv1d_32/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_32/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_32/conv1d/ExpandDims_1/dim�
conv1d_32/conv1d/ExpandDims_1
ExpandDims4conv1d_32/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_32/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_32/conv1d/ExpandDims_1�
conv1d_32/conv1dConv2D$conv1d_32/conv1d/ExpandDims:output:0&conv1d_32/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_32/conv1d�
conv1d_32/conv1d/SqueezeSqueezeconv1d_32/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_32/conv1d/Squeeze�
 conv1d_32/BiasAdd/ReadVariableOpReadVariableOp)conv1d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_32/BiasAdd/ReadVariableOp�
conv1d_32/BiasAddBiasAdd!conv1d_32/conv1d/Squeeze:output:0(conv1d_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_32/BiasAddz
conv1d_32/ReluReluconv1d_32/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_32/Relu�
conv1d_33/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_33/conv1d/ExpandDims/dim�
conv1d_33/conv1d/ExpandDims
ExpandDimsconv1d_32/Relu:activations:0(conv1d_33/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d_33/conv1d/ExpandDims�
,conv1d_33/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_33_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02.
,conv1d_33/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_33/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_33/conv1d/ExpandDims_1/dim�
conv1d_33/conv1d/ExpandDims_1
ExpandDims4conv1d_33/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_33/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_33/conv1d/ExpandDims_1�
conv1d_33/conv1dConv2D$conv1d_33/conv1d/ExpandDims:output:0&conv1d_33/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_33/conv1d�
conv1d_33/conv1d/SqueezeSqueezeconv1d_33/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_33/conv1d/Squeeze�
 conv1d_33/BiasAdd/ReadVariableOpReadVariableOp)conv1d_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_33/BiasAdd/ReadVariableOp�
conv1d_33/BiasAddBiasAdd!conv1d_33/conv1d/Squeeze:output:0(conv1d_33/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_33/BiasAddz
conv1d_33/ReluReluconv1d_33/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_33/Reluy
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_16/dropout/Const�
dropout_16/dropout/MulMulconv1d_33/Relu:activations:0!dropout_16/dropout/Const:output:0*
T0*+
_output_shapes
:��������� 2
dropout_16/dropout/Mul�
dropout_16/dropout/ShapeShapeconv1d_33/Relu:activations:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape�
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*+
_output_shapes
:��������� *
dtype021
/dropout_16/dropout/random_uniform/RandomUniform�
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_16/dropout/GreaterEqual/y�
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:��������� 2!
dropout_16/dropout/GreaterEqual�
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:��������� 2
dropout_16/dropout/Cast�
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*+
_output_shapes
:��������� 2
dropout_16/dropout/Mul_1�
max_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_16/ExpandDims/dim�
max_pooling1d_16/ExpandDims
ExpandDimsdropout_16/dropout/Mul_1:z:0(max_pooling1d_16/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
max_pooling1d_16/ExpandDims�
max_pooling1d_16/MaxPoolMaxPool$max_pooling1d_16/ExpandDims:output:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling1d_16/MaxPool�
max_pooling1d_16/SqueezeSqueeze!max_pooling1d_16/MaxPool:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims
2
max_pooling1d_16/Squeezeu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_16/Const�
flatten_16/ReshapeReshape!max_pooling1d_16/Squeeze:output:0flatten_16/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_16/Reshape�
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02 
dense_32/MatMul/ReadVariableOp�
dense_32/MatMulMatMulflatten_16/Reshape:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_32/MatMul�
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_32/BiasAdd/ReadVariableOp�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_32/BiasAdds
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
dense_32/Relu�
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_33/MatMul/ReadVariableOp�
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_33/MatMul�
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_33/BiasAdd/ReadVariableOp�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_33/BiasAdd|
dense_33/SoftmaxSoftmaxdense_33/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_33/Softmaxn
IdentityIdentitydense_33/Softmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������:::::::::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_800685

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:��������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:��������� 2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:��������� 2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�!
�
I__inference_sequential_16_layer_call_and_return_conditional_losses_800311
conv1d_32_input
conv1d_32_800174
conv1d_32_800176
conv1d_33_800206
conv1d_33_800208
dense_32_800278
dense_32_800280
dense_33_800305
dense_33_800307
identity��!conv1d_32/StatefulPartitionedCall�!conv1d_33/StatefulPartitionedCall� dense_32/StatefulPartitionedCall� dense_33/StatefulPartitionedCall�"dropout_16/StatefulPartitionedCall�
!conv1d_32/StatefulPartitionedCallStatefulPartitionedCallconv1d_32_inputconv1d_32_800174conv1d_32_800176*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_32_layer_call_and_return_conditional_losses_8001632#
!conv1d_32/StatefulPartitionedCall�
!conv1d_33/StatefulPartitionedCallStatefulPartitionedCall*conv1d_32/StatefulPartitionedCall:output:0conv1d_33_800206conv1d_33_800208*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_33_layer_call_and_return_conditional_losses_8001982#
!conv1d_33/StatefulPartitionedCall�
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall*conv1d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_8002262$
"dropout_16/StatefulPartitionedCall�
 max_pooling1d_16/PartitionedCallPartitionedCall+dropout_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_8001372"
 max_pooling1d_16/PartitionedCall�
flatten_16/PartitionedCallPartitionedCall)max_pooling1d_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_8002512
flatten_16/PartitionedCall�
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_800278dense_32_800280*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_8002702"
 dense_32/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_800305dense_33_800307*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_8002942"
 dense_33/StatefulPartitionedCall�
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0"^conv1d_32/StatefulPartitionedCall"^conv1d_33/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_32/StatefulPartitionedCall!conv1d_32/StatefulPartitionedCall2F
!conv1d_33/StatefulPartitionedCall!conv1d_33/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_32_input
�
�
E__inference_conv1d_33_layer_call_and_return_conditional_losses_800198

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :::S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
� 
�
I__inference_sequential_16_layer_call_and_return_conditional_losses_800338
conv1d_32_input
conv1d_32_800314
conv1d_32_800316
conv1d_33_800319
conv1d_33_800321
dense_32_800327
dense_32_800329
dense_33_800332
dense_33_800334
identity��!conv1d_32/StatefulPartitionedCall�!conv1d_33/StatefulPartitionedCall� dense_32/StatefulPartitionedCall� dense_33/StatefulPartitionedCall�
!conv1d_32/StatefulPartitionedCallStatefulPartitionedCallconv1d_32_inputconv1d_32_800314conv1d_32_800316*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_32_layer_call_and_return_conditional_losses_8001632#
!conv1d_32/StatefulPartitionedCall�
!conv1d_33/StatefulPartitionedCallStatefulPartitionedCall*conv1d_32/StatefulPartitionedCall:output:0conv1d_33_800319conv1d_33_800321*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_33_layer_call_and_return_conditional_losses_8001982#
!conv1d_33/StatefulPartitionedCall�
dropout_16/PartitionedCallPartitionedCall*conv1d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_8002362
dropout_16/PartitionedCall�
 max_pooling1d_16/PartitionedCallPartitionedCall#dropout_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_8001372"
 max_pooling1d_16/PartitionedCall�
flatten_16/PartitionedCallPartitionedCall)max_pooling1d_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_8002512
flatten_16/PartitionedCall�
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_800327dense_32_800329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_8002702"
 dense_32/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_800332dense_33_800334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_8002942"
 dense_33/StatefulPartitionedCall�
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0"^conv1d_32/StatefulPartitionedCall"^conv1d_33/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_32/StatefulPartitionedCall!conv1d_32/StatefulPartitionedCall2F
!conv1d_33/StatefulPartitionedCall!conv1d_33/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_32_input
�H
�
__inference__traced_save_800863
file_prefix/
+savev2_conv1d_32_kernel_read_readvariableop-
)savev2_conv1d_32_bias_read_readvariableop/
+savev2_conv1d_33_kernel_read_readvariableop-
)savev2_conv1d_33_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv1d_32_kernel_m_read_readvariableop4
0savev2_adam_conv1d_32_bias_m_read_readvariableop6
2savev2_adam_conv1d_33_kernel_m_read_readvariableop4
0savev2_adam_conv1d_33_bias_m_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableop5
1savev2_adam_dense_33_kernel_m_read_readvariableop3
/savev2_adam_dense_33_bias_m_read_readvariableop6
2savev2_adam_conv1d_32_kernel_v_read_readvariableop4
0savev2_adam_conv1d_32_bias_v_read_readvariableop6
2savev2_adam_conv1d_33_kernel_v_read_readvariableop4
0savev2_adam_conv1d_33_bias_v_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableop5
1savev2_adam_dense_33_kernel_v_read_readvariableop3
/savev2_adam_dense_33_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1b3d0f41ca6e4a46bfbd0b0460962c4c/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*�
value�B�"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_32_kernel_read_readvariableop)savev2_conv1d_32_bias_read_readvariableop+savev2_conv1d_33_kernel_read_readvariableop)savev2_conv1d_33_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv1d_32_kernel_m_read_readvariableop0savev2_adam_conv1d_32_bias_m_read_readvariableop2savev2_adam_conv1d_33_kernel_m_read_readvariableop0savev2_adam_conv1d_33_bias_m_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableop1savev2_adam_dense_33_kernel_m_read_readvariableop/savev2_adam_dense_33_bias_m_read_readvariableop2savev2_adam_conv1d_32_kernel_v_read_readvariableop0savev2_adam_conv1d_32_bias_v_read_readvariableop2savev2_adam_conv1d_33_kernel_v_read_readvariableop0savev2_adam_conv1d_33_bias_v_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableop1savev2_adam_dense_33_kernel_v_read_readvariableop/savev2_adam_dense_33_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : :	�d:d:d:: : : : : : : : : : : :  : :	�d:d:d:: : :  : :	�d:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :%!

_output_shapes
:	�d: 

_output_shapes
:d:$  

_output_shapes

:d: !

_output_shapes
::"

_output_shapes
: 
�
�
.__inference_sequential_16_layer_call_fn_800435
conv1d_32_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_32_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_8004162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_32_input
�
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_800236

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:��������� 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
O
conv1d_32_input<
!serving_default_conv1d_32_input:0���������<
dense_330
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�:
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		keras_api

regularization_losses

signatures
	variables
trainable_variables
z_default_save_signature
{__call__
*|&call_and_return_all_conditional_losses"�6
_tf_keras_sequential�6{"training_config": {"weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"beta_1": 0.8999999761581421, "epsilon": 1e-07, "beta_2": 0.9990000128746033, "learning_rate": 9.999999747378752e-05, "decay": 0.0, "name": "Adam", "amsgrad": false}}, "loss": "categorical_crossentropy", "metrics": ["accuracy"], "loss_weights": null}, "expects_training_arg": true, "model_config": {"class_name": "Sequential", "config": {"layers": [{"class_name": "InputLayer", "config": {"sparse": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "dtype": "float32", "ragged": false, "name": "conv1d_32_input"}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "name": "conv1d_32", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "name": "conv1d_33", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Dropout", "config": {"dtype": "float32", "seed": null, "rate": 0.5, "noise_shape": null, "name": "dropout_16", "trainable": true}}, {"class_name": "MaxPooling1D", "config": {"strides": {"class_name": "__tuple__", "items": [2]}, "name": "max_pooling1d_16", "data_format": "channels_last", "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "trainable": true, "dtype": "float32"}}, {"class_name": "Flatten", "config": {"dtype": "float32", "name": "flatten_16", "data_format": "channels_last", "trainable": true}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 100, "kernel_constraint": null, "activity_regularizer": null, "activation": "relu", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_32", "trainable": true, "kernel_regularizer": null}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 2, "kernel_constraint": null, "activity_regularizer": null, "activation": "softmax", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_33", "trainable": true, "kernel_regularizer": null}}], "name": "sequential_16"}}, "class_name": "Sequential", "must_restore_from_config": false, "dtype": "float32", "config": {"layers": [{"class_name": "InputLayer", "config": {"sparse": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "dtype": "float32", "ragged": false, "name": "conv1d_32_input"}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "name": "conv1d_32", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "name": "conv1d_33", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Dropout", "config": {"dtype": "float32", "seed": null, "rate": 0.5, "noise_shape": null, "name": "dropout_16", "trainable": true}}, {"class_name": "MaxPooling1D", "config": {"strides": {"class_name": "__tuple__", "items": [2]}, "name": "max_pooling1d_16", "data_format": "channels_last", "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "trainable": true, "dtype": "float32"}}, {"class_name": "Flatten", "config": {"dtype": "float32", "name": "flatten_16", "data_format": "channels_last", "trainable": true}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 100, "kernel_constraint": null, "activity_regularizer": null, "activation": "relu", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_32", "trainable": true, "kernel_regularizer": null}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 2, "kernel_constraint": null, "activity_regularizer": null, "activation": "softmax", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_33", "trainable": true, "kernel_regularizer": null}}], "name": "sequential_16"}, "keras_version": "2.4.0", "batch_input_shape": null, "is_graph_network": true, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 5}, "max_ndim": null, "min_ndim": 3}}, "name": "sequential_16", "backend": "tensorflow", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 5]}}
�


kernel
bias
	keras_api
regularization_losses
	variables
trainable_variables
}__call__
*~&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"config": {"dtype": "float32", "use_bias": true, "strides": {"class_name": "__tuple__", "items": [1]}, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "activity_regularizer": null, "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "groups": 1, "filters": 32, "bias_regularizer": null, "activation": "relu", "name": "conv1d_32", "kernel_constraint": null, "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "name": "conv1d_32", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 5}, "max_ndim": null, "min_ndim": 3}}, "class_name": "Conv1D", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 5]}, "trainable": true}
�	

kernel
bias
	keras_api
regularization_losses
	variables
trainable_variables
__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"config": {"dtype": "float32", "use_bias": true, "strides": {"class_name": "__tuple__", "items": [1]}, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "activity_regularizer": null, "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "groups": 1, "filters": 32, "bias_regularizer": null, "activation": "relu", "name": "conv1d_33", "kernel_constraint": null, "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": null, "name": "conv1d_33", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 32}, "max_ndim": null, "min_ndim": 3}}, "class_name": "Conv1D", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 32]}, "trainable": true}
�
	keras_api
regularization_losses
	variables
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"must_restore_from_config": false, "batch_input_shape": null, "name": "dropout_16", "config": {"trainable": true, "seed": null, "rate": 0.5, "noise_shape": null, "name": "dropout_16", "dtype": "float32"}, "class_name": "Dropout", "expects_training_arg": true, "dtype": "float32", "stateful": false, "trainable": true}
�
	keras_api
regularization_losses
 	variables
!trainable_variables
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"config": {"trainable": true, "strides": {"class_name": "__tuple__", "items": [2]}, "data_format": "channels_last", "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "dtype": "float32", "name": "max_pooling1d_16"}, "must_restore_from_config": false, "batch_input_shape": null, "name": "max_pooling1d_16", "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "axes": {}, "max_ndim": null, "min_ndim": null}}, "class_name": "MaxPooling1D", "expects_training_arg": false, "dtype": "float32", "stateful": false, "trainable": true}
�
"	keras_api
#regularization_losses
$	variables
%trainable_variables
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"config": {"name": "flatten_16", "trainable": true, "data_format": "channels_last", "dtype": "float32"}, "must_restore_from_config": false, "batch_input_shape": null, "name": "flatten_16", "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {}, "max_ndim": null, "min_ndim": 1}}, "class_name": "Flatten", "expects_training_arg": false, "dtype": "float32", "stateful": false, "trainable": true}
�

&kernel
'bias
(	keras_api
)regularization_losses
*	variables
+trainable_variables
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"config": {"dtype": "float32", "use_bias": true, "units": 100, "kernel_constraint": null, "activity_regularizer": null, "activation": "relu", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_32", "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": null, "name": "dense_32", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 192}, "max_ndim": null, "min_ndim": 2}}, "class_name": "Dense", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 192]}, "trainable": true}
�

,kernel
-bias
.	keras_api
/regularization_losses
0	variables
1trainable_variables
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"config": {"dtype": "float32", "use_bias": true, "units": 2, "kernel_constraint": null, "activity_regularizer": null, "activation": "softmax", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_33", "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": null, "name": "dense_33", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 100}, "max_ndim": null, "min_ndim": 2}}, "class_name": "Dense", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "trainable": true}
�
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy"
	optimizer
�

7layers
8non_trainable_variables
trainable_variables
9layer_metrics
	variables

regularization_losses
:layer_regularization_losses
;metrics
z_default_save_signature
&|"call_and_return_conditional_losses
{__call__
*|&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
-
�serving_default"
signature_map
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
&:$ 2conv1d_32/kernel
: 2conv1d_32/bias
�
<non_trainable_variables
trainable_variables
=layer_metrics
regularization_losses
	variables

>layers
?layer_regularization_losses
@metrics
&~"call_and_return_conditional_losses
}__call__
*~&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
&:$  2conv1d_33/kernel
: 2conv1d_33/bias
�
Anon_trainable_variables
trainable_variables
Blayer_metrics
regularization_losses
	variables

Clayers
Dlayer_regularization_losses
Emetrics
'�"call_and_return_conditional_losses
__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Fnon_trainable_variables
trainable_variables
Glayer_metrics
regularization_losses
	variables

Hlayers
Ilayer_regularization_losses
Jmetrics
'�"call_and_return_conditional_losses
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Knon_trainable_variables
!trainable_variables
Llayer_metrics
regularization_losses
 	variables

Mlayers
Nlayer_regularization_losses
Ometrics
'�"call_and_return_conditional_losses
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Pnon_trainable_variables
%trainable_variables
Qlayer_metrics
#regularization_losses
$	variables

Rlayers
Slayer_regularization_losses
Tmetrics
'�"call_and_return_conditional_losses
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
": 	�d2dense_32/kernel
:d2dense_32/bias
�
Unon_trainable_variables
+trainable_variables
Vlayer_metrics
)regularization_losses
*	variables

Wlayers
Xlayer_regularization_losses
Ymetrics
'�"call_and_return_conditional_losses
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
!:d2dense_33/kernel
:2dense_33/bias
�
Znon_trainable_variables
1trainable_variables
[layer_metrics
/regularization_losses
0	variables

\layers
]layer_regularization_losses
^metrics
'�"call_and_return_conditional_losses
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	atotal
	bcount
c	keras_api
d	variables"�
_tf_keras_metricj{"class_name": "Mean", "config": {"name": "loss", "dtype": "float32"}, "dtype": "float32", "name": "loss"}
�
	etotal
	fcount
g
_fn_kwargs
h	keras_api
i	variables"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "config": {"fn": "categorical_accuracy", "name": "accuracy", "dtype": "float32"}, "dtype": "float32", "name": "accuracy"}
:  (2total
:  (2count
-
d	variables"
_generic_user_object
.
a0
b1"
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
-
i	variables"
_generic_user_object
.
e0
f1"
trackable_list_wrapper
+:) 2Adam/conv1d_32/kernel/m
!: 2Adam/conv1d_32/bias/m
+:)  2Adam/conv1d_33/kernel/m
!: 2Adam/conv1d_33/bias/m
':%	�d2Adam/dense_32/kernel/m
 :d2Adam/dense_32/bias/m
&:$d2Adam/dense_33/kernel/m
 :2Adam/dense_33/bias/m
+:) 2Adam/conv1d_32/kernel/v
!: 2Adam/conv1d_32/bias/v
+:)  2Adam/conv1d_33/kernel/v
!: 2Adam/conv1d_33/bias/v
':%	�d2Adam/dense_32/kernel/v
 :d2Adam/dense_32/bias/v
&:$d2Adam/dense_33/kernel/v
 :2Adam/dense_33/bias/v
�2�
!__inference__wrapped_model_800128�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *2�/
-�*
conv1d_32_input���������
�2�
.__inference_sequential_16_layer_call_fn_800487
.__inference_sequential_16_layer_call_fn_800435
.__inference_sequential_16_layer_call_fn_800387
.__inference_sequential_16_layer_call_fn_800508�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
I__inference_sequential_16_layer_call_and_return_conditional_losses_800564
I__inference_sequential_16_layer_call_and_return_conditional_losses_800613
I__inference_sequential_16_layer_call_and_return_conditional_losses_800311
I__inference_sequential_16_layer_call_and_return_conditional_losses_800338�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_conv1d_32_layer_call_fn_800638�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv1d_32_layer_call_and_return_conditional_losses_800629�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv1d_33_layer_call_fn_800647�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv1d_33_layer_call_and_return_conditional_losses_800663�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dropout_16_layer_call_fn_800673
+__inference_dropout_16_layer_call_fn_800668�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_16_layer_call_and_return_conditional_losses_800690
F__inference_dropout_16_layer_call_and_return_conditional_losses_800685�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
1__inference_max_pooling1d_16_layer_call_fn_800143�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
L__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_800137�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
+__inference_flatten_16_layer_call_fn_800695�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_flatten_16_layer_call_and_return_conditional_losses_800701�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_32_layer_call_fn_800710�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_32_layer_call_and_return_conditional_losses_800721�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_33_layer_call_fn_800741�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_33_layer_call_and_return_conditional_losses_800732�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
;B9
$__inference_signature_wrapper_800466conv1d_32_input�
!__inference__wrapped_model_800128}&',-<�9
2�/
-�*
conv1d_32_input���������
� "3�0
.
dense_33"�
dense_33����������
E__inference_conv1d_32_layer_call_and_return_conditional_losses_800629d3�0
)�&
$�!
inputs���������
� ")�&
�
0��������� 
� �
*__inference_conv1d_32_layer_call_fn_800638W3�0
)�&
$�!
inputs���������
� "���������� �
E__inference_conv1d_33_layer_call_and_return_conditional_losses_800663d3�0
)�&
$�!
inputs��������� 
� ")�&
�
0��������� 
� �
*__inference_conv1d_33_layer_call_fn_800647W3�0
)�&
$�!
inputs��������� 
� "���������� �
D__inference_dense_32_layer_call_and_return_conditional_losses_800721]&'0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� }
)__inference_dense_32_layer_call_fn_800710P&'0�-
&�#
!�
inputs����������
� "����������d�
D__inference_dense_33_layer_call_and_return_conditional_losses_800732\,-/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� |
)__inference_dense_33_layer_call_fn_800741O,-/�,
%�"
 �
inputs���������d
� "�����������
F__inference_dropout_16_layer_call_and_return_conditional_losses_800685d7�4
-�*
$�!
inputs��������� 
p
� ")�&
�
0��������� 
� �
F__inference_dropout_16_layer_call_and_return_conditional_losses_800690d7�4
-�*
$�!
inputs��������� 
p 
� ")�&
�
0��������� 
� �
+__inference_dropout_16_layer_call_fn_800668W7�4
-�*
$�!
inputs��������� 
p
� "���������� �
+__inference_dropout_16_layer_call_fn_800673W7�4
-�*
$�!
inputs��������� 
p 
� "���������� �
F__inference_flatten_16_layer_call_and_return_conditional_losses_800701]3�0
)�&
$�!
inputs��������� 
� "&�#
�
0����������
� 
+__inference_flatten_16_layer_call_fn_800695P3�0
)�&
$�!
inputs��������� 
� "������������
L__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_800137�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
1__inference_max_pooling1d_16_layer_call_fn_800143wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
I__inference_sequential_16_layer_call_and_return_conditional_losses_800311w&',-D�A
:�7
-�*
conv1d_32_input���������
p

 
� "%�"
�
0���������
� �
I__inference_sequential_16_layer_call_and_return_conditional_losses_800338w&',-D�A
:�7
-�*
conv1d_32_input���������
p 

 
� "%�"
�
0���������
� �
I__inference_sequential_16_layer_call_and_return_conditional_losses_800564n&',-;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
I__inference_sequential_16_layer_call_and_return_conditional_losses_800613n&',-;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
.__inference_sequential_16_layer_call_fn_800387j&',-D�A
:�7
-�*
conv1d_32_input���������
p

 
� "�����������
.__inference_sequential_16_layer_call_fn_800435j&',-D�A
:�7
-�*
conv1d_32_input���������
p 

 
� "�����������
.__inference_sequential_16_layer_call_fn_800487a&',-;�8
1�.
$�!
inputs���������
p

 
� "�����������
.__inference_sequential_16_layer_call_fn_800508a&',-;�8
1�.
$�!
inputs���������
p 

 
� "�����������
$__inference_signature_wrapper_800466�&',-O�L
� 
E�B
@
conv1d_32_input-�*
conv1d_32_input���������"3�0
.
dense_33"�
dense_33���������