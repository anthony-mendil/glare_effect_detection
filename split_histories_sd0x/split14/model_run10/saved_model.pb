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
conv1d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv1d_20/kernel
y
$conv1d_20/kernel/Read/ReadVariableOpReadVariableOpconv1d_20/kernel*"
_output_shapes
: *
dtype0
t
conv1d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_20/bias
m
"conv1d_20/bias/Read/ReadVariableOpReadVariableOpconv1d_20/bias*
_output_shapes
: *
dtype0
�
conv1d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv1d_21/kernel
y
$conv1d_21/kernel/Read/ReadVariableOpReadVariableOpconv1d_21/kernel*"
_output_shapes
:  *
dtype0
t
conv1d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_21/bias
m
"conv1d_21/bias/Read/ReadVariableOpReadVariableOpconv1d_21/bias*
_output_shapes
: *
dtype0
{
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d* 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes
:	�d*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:d*
dtype0
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:d*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
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
Adam/conv1d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv1d_20/kernel/m
�
+Adam/conv1d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_20/kernel/m*"
_output_shapes
: *
dtype0
�
Adam/conv1d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv1d_20/bias/m
{
)Adam/conv1d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_20/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv1d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv1d_21/kernel/m
�
+Adam/conv1d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_21/kernel/m*"
_output_shapes
:  *
dtype0
�
Adam/conv1d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv1d_21/bias/m
{
)Adam/conv1d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_21/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*'
shared_nameAdam/dense_20/kernel/m
�
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes
:	�d*
dtype0
�
Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:d*
dtype0
�
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_21/kernel/m
�
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes

:d*
dtype0
�
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/m
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv1d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv1d_20/kernel/v
�
+Adam/conv1d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_20/kernel/v*"
_output_shapes
: *
dtype0
�
Adam/conv1d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv1d_20/bias/v
{
)Adam/conv1d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_20/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv1d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv1d_21/kernel/v
�
+Adam/conv1d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_21/kernel/v*"
_output_shapes
:  *
dtype0
�
Adam/conv1d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv1d_21/bias/v
{
)Adam/conv1d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_21/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*'
shared_nameAdam/dense_20/kernel/v
�
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes
:	�d*
dtype0
�
Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:d*
dtype0
�
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_21/kernel/v
�
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes

:d*
dtype0
�
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/v
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
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
VARIABLE_VALUEconv1d_20/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_20/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv1d_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_20/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_21/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_21/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/conv1d_20/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_20/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_21/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_21/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_20/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_20/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_21/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_21/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_conv1d_20_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_20_inputconv1d_20/kernelconv1d_20/biasconv1d_21/kernelconv1d_21/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/bias*
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
$__inference_signature_wrapper_517728
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_20/kernel/Read/ReadVariableOp"conv1d_20/bias/Read/ReadVariableOp$conv1d_21/kernel/Read/ReadVariableOp"conv1d_21/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv1d_20/kernel/m/Read/ReadVariableOp)Adam/conv1d_20/bias/m/Read/ReadVariableOp+Adam/conv1d_21/kernel/m/Read/ReadVariableOp)Adam/conv1d_21/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp+Adam/conv1d_20/kernel/v/Read/ReadVariableOp)Adam/conv1d_20/bias/v/Read/ReadVariableOp+Adam/conv1d_21/kernel/v/Read/ReadVariableOp)Adam/conv1d_21/bias/v/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOpConst*.
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
__inference__traced_save_518125
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_20/kernelconv1d_20/biasconv1d_21/kernelconv1d_21/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv1d_20/kernel/mAdam/conv1d_20/bias/mAdam/conv1d_21/kernel/mAdam/conv1d_21/bias/mAdam/dense_20/kernel/mAdam/dense_20/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/conv1d_20/kernel/vAdam/conv1d_20/bias/vAdam/conv1d_21/kernel/vAdam/conv1d_21/bias/vAdam/dense_20/kernel/vAdam/dense_20/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/v*-
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
"__inference__traced_restore_518234ր
�
�
D__inference_dense_20_layer_call_and_return_conditional_losses_517529

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
�
d
+__inference_dropout_10_layer_call_fn_517930

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
F__inference_dropout_10_layer_call_and_return_conditional_losses_5174882
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
�
�
$__inference_signature_wrapper_517728
conv1d_20_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
!__inference__wrapped_model_5173902
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
_user_specified_nameconv1d_20_input
�
~
)__inference_dense_21_layer_call_fn_517992

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
D__inference_dense_21_layer_call_and_return_conditional_losses_5175592
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
�
�
D__inference_dense_21_layer_call_and_return_conditional_losses_517559

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
D__inference_dense_21_layer_call_and_return_conditional_losses_518003

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
.__inference_sequential_10_layer_call_fn_517697
conv1d_20_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
I__inference_sequential_10_layer_call_and_return_conditional_losses_5176782
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
_user_specified_nameconv1d_20_input
�
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_517488

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
�
�
E__inference_conv1d_20_layer_call_and_return_conditional_losses_517891

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
�
�
E__inference_conv1d_20_layer_call_and_return_conditional_losses_517425

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
�
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_517952

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
�H
�
__inference__traced_save_518125
file_prefix/
+savev2_conv1d_20_kernel_read_readvariableop-
)savev2_conv1d_20_bias_read_readvariableop/
+savev2_conv1d_21_kernel_read_readvariableop-
)savev2_conv1d_21_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv1d_20_kernel_m_read_readvariableop4
0savev2_adam_conv1d_20_bias_m_read_readvariableop6
2savev2_adam_conv1d_21_kernel_m_read_readvariableop4
0savev2_adam_conv1d_21_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop6
2savev2_adam_conv1d_20_kernel_v_read_readvariableop4
0savev2_adam_conv1d_20_bias_v_read_readvariableop6
2savev2_adam_conv1d_21_kernel_v_read_readvariableop4
0savev2_adam_conv1d_21_bias_v_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop
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
value3B1 B+_temp_b01b6f2300d4474d8deed54df3f19c76/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_20_kernel_read_readvariableop)savev2_conv1d_20_bias_read_readvariableop+savev2_conv1d_21_kernel_read_readvariableop)savev2_conv1d_21_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv1d_20_kernel_m_read_readvariableop0savev2_adam_conv1d_20_bias_m_read_readvariableop2savev2_adam_conv1d_21_kernel_m_read_readvariableop0savev2_adam_conv1d_21_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop2savev2_adam_conv1d_20_kernel_v_read_readvariableop0savev2_adam_conv1d_20_bias_v_read_readvariableop2savev2_adam_conv1d_21_kernel_v_read_readvariableop0savev2_adam_conv1d_21_bias_v_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�E
�
!__inference__wrapped_model_517390
conv1d_20_inputG
Csequential_10_conv1d_20_conv1d_expanddims_1_readvariableop_resource;
7sequential_10_conv1d_20_biasadd_readvariableop_resourceG
Csequential_10_conv1d_21_conv1d_expanddims_1_readvariableop_resource;
7sequential_10_conv1d_21_biasadd_readvariableop_resource9
5sequential_10_dense_20_matmul_readvariableop_resource:
6sequential_10_dense_20_biasadd_readvariableop_resource9
5sequential_10_dense_21_matmul_readvariableop_resource:
6sequential_10_dense_21_biasadd_readvariableop_resource
identity��
-sequential_10/conv1d_20/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_10/conv1d_20/conv1d/ExpandDims/dim�
)sequential_10/conv1d_20/conv1d/ExpandDims
ExpandDimsconv1d_20_input6sequential_10/conv1d_20/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2+
)sequential_10/conv1d_20/conv1d/ExpandDims�
:sequential_10/conv1d_20/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_10_conv1d_20_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02<
:sequential_10/conv1d_20/conv1d/ExpandDims_1/ReadVariableOp�
/sequential_10/conv1d_20/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_10/conv1d_20/conv1d/ExpandDims_1/dim�
+sequential_10/conv1d_20/conv1d/ExpandDims_1
ExpandDimsBsequential_10/conv1d_20/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_10/conv1d_20/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2-
+sequential_10/conv1d_20/conv1d/ExpandDims_1�
sequential_10/conv1d_20/conv1dConv2D2sequential_10/conv1d_20/conv1d/ExpandDims:output:04sequential_10/conv1d_20/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2 
sequential_10/conv1d_20/conv1d�
&sequential_10/conv1d_20/conv1d/SqueezeSqueeze'sequential_10/conv1d_20/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2(
&sequential_10/conv1d_20/conv1d/Squeeze�
.sequential_10/conv1d_20/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv1d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_10/conv1d_20/BiasAdd/ReadVariableOp�
sequential_10/conv1d_20/BiasAddBiasAdd/sequential_10/conv1d_20/conv1d/Squeeze:output:06sequential_10/conv1d_20/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2!
sequential_10/conv1d_20/BiasAdd�
sequential_10/conv1d_20/ReluRelu(sequential_10/conv1d_20/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
sequential_10/conv1d_20/Relu�
-sequential_10/conv1d_21/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_10/conv1d_21/conv1d/ExpandDims/dim�
)sequential_10/conv1d_21/conv1d/ExpandDims
ExpandDims*sequential_10/conv1d_20/Relu:activations:06sequential_10/conv1d_21/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2+
)sequential_10/conv1d_21/conv1d/ExpandDims�
:sequential_10/conv1d_21/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_10_conv1d_21_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02<
:sequential_10/conv1d_21/conv1d/ExpandDims_1/ReadVariableOp�
/sequential_10/conv1d_21/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_10/conv1d_21/conv1d/ExpandDims_1/dim�
+sequential_10/conv1d_21/conv1d/ExpandDims_1
ExpandDimsBsequential_10/conv1d_21/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_10/conv1d_21/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2-
+sequential_10/conv1d_21/conv1d/ExpandDims_1�
sequential_10/conv1d_21/conv1dConv2D2sequential_10/conv1d_21/conv1d/ExpandDims:output:04sequential_10/conv1d_21/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2 
sequential_10/conv1d_21/conv1d�
&sequential_10/conv1d_21/conv1d/SqueezeSqueeze'sequential_10/conv1d_21/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2(
&sequential_10/conv1d_21/conv1d/Squeeze�
.sequential_10/conv1d_21/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv1d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_10/conv1d_21/BiasAdd/ReadVariableOp�
sequential_10/conv1d_21/BiasAddBiasAdd/sequential_10/conv1d_21/conv1d/Squeeze:output:06sequential_10/conv1d_21/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2!
sequential_10/conv1d_21/BiasAdd�
sequential_10/conv1d_21/ReluRelu(sequential_10/conv1d_21/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
sequential_10/conv1d_21/Relu�
!sequential_10/dropout_10/IdentityIdentity*sequential_10/conv1d_21/Relu:activations:0*
T0*+
_output_shapes
:��������� 2#
!sequential_10/dropout_10/Identity�
-sequential_10/max_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_10/max_pooling1d_10/ExpandDims/dim�
)sequential_10/max_pooling1d_10/ExpandDims
ExpandDims*sequential_10/dropout_10/Identity:output:06sequential_10/max_pooling1d_10/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2+
)sequential_10/max_pooling1d_10/ExpandDims�
&sequential_10/max_pooling1d_10/MaxPoolMaxPool2sequential_10/max_pooling1d_10/ExpandDims:output:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2(
&sequential_10/max_pooling1d_10/MaxPool�
&sequential_10/max_pooling1d_10/SqueezeSqueeze/sequential_10/max_pooling1d_10/MaxPool:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims
2(
&sequential_10/max_pooling1d_10/Squeeze�
sequential_10/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2 
sequential_10/flatten_10/Const�
 sequential_10/flatten_10/ReshapeReshape/sequential_10/max_pooling1d_10/Squeeze:output:0'sequential_10/flatten_10/Const:output:0*
T0*(
_output_shapes
:����������2"
 sequential_10/flatten_10/Reshape�
,sequential_10/dense_20/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_20_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02.
,sequential_10/dense_20/MatMul/ReadVariableOp�
sequential_10/dense_20/MatMulMatMul)sequential_10/flatten_10/Reshape:output:04sequential_10/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential_10/dense_20/MatMul�
-sequential_10/dense_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_10/dense_20/BiasAdd/ReadVariableOp�
sequential_10/dense_20/BiasAddBiasAdd'sequential_10/dense_20/MatMul:product:05sequential_10/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2 
sequential_10/dense_20/BiasAdd�
sequential_10/dense_20/ReluRelu'sequential_10/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
sequential_10/dense_20/Relu�
,sequential_10/dense_21/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_21_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_10/dense_21/MatMul/ReadVariableOp�
sequential_10/dense_21/MatMulMatMul)sequential_10/dense_20/Relu:activations:04sequential_10/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_10/dense_21/MatMul�
-sequential_10/dense_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_21/BiasAdd/ReadVariableOp�
sequential_10/dense_21/BiasAddBiasAdd'sequential_10/dense_21/MatMul:product:05sequential_10/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_10/dense_21/BiasAdd�
sequential_10/dense_21/SoftmaxSoftmax'sequential_10/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������2 
sequential_10/dense_21/Softmax|
IdentityIdentity(sequential_10/dense_21/Softmax:softmax:0*
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
_user_specified_nameconv1d_20_input
�
�
.__inference_sequential_10_layer_call_fn_517875

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
I__inference_sequential_10_layer_call_and_return_conditional_losses_5176782
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
�
~
)__inference_dense_20_layer_call_fn_517983

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
D__inference_dense_20_layer_call_and_return_conditional_losses_5175292
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
�
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_517498

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
�

*__inference_conv1d_20_layer_call_fn_517900

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
E__inference_conv1d_20_layer_call_and_return_conditional_losses_5174252
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
.__inference_sequential_10_layer_call_fn_517649
conv1d_20_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
I__inference_sequential_10_layer_call_and_return_conditional_losses_5176302
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
_user_specified_nameconv1d_20_input
��
�
"__inference__traced_restore_518234
file_prefix%
!assignvariableop_conv1d_20_kernel%
!assignvariableop_1_conv1d_20_bias'
#assignvariableop_2_conv1d_21_kernel%
!assignvariableop_3_conv1d_21_bias&
"assignvariableop_4_dense_20_kernel$
 assignvariableop_5_dense_20_bias&
"assignvariableop_6_dense_21_kernel$
 assignvariableop_7_dense_21_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1/
+assignvariableop_17_adam_conv1d_20_kernel_m-
)assignvariableop_18_adam_conv1d_20_bias_m/
+assignvariableop_19_adam_conv1d_21_kernel_m-
)assignvariableop_20_adam_conv1d_21_bias_m.
*assignvariableop_21_adam_dense_20_kernel_m,
(assignvariableop_22_adam_dense_20_bias_m.
*assignvariableop_23_adam_dense_21_kernel_m,
(assignvariableop_24_adam_dense_21_bias_m/
+assignvariableop_25_adam_conv1d_20_kernel_v-
)assignvariableop_26_adam_conv1d_20_bias_v/
+assignvariableop_27_adam_conv1d_21_kernel_v-
)assignvariableop_28_adam_conv1d_21_bias_v.
*assignvariableop_29_adam_dense_20_kernel_v,
(assignvariableop_30_adam_dense_20_bias_v.
*assignvariableop_31_adam_dense_21_kernel_v,
(assignvariableop_32_adam_dense_21_bias_v
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
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_20_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_20_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_21_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_21_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_20_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_20_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_21_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_21_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv1d_20_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv1d_20_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv1d_21_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv1d_21_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_20_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_20_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_21_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_21_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv1d_20_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv1d_20_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv1d_21_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv1d_21_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_20_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_20_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_21_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_21_bias_vIdentity_32:output:0"/device:CPU:0*
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
�
b
F__inference_flatten_10_layer_call_and_return_conditional_losses_517958

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
�A
�
I__inference_sequential_10_layer_call_and_return_conditional_losses_517784

inputs9
5conv1d_20_conv1d_expanddims_1_readvariableop_resource-
)conv1d_20_biasadd_readvariableop_resource9
5conv1d_21_conv1d_expanddims_1_readvariableop_resource-
)conv1d_21_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource
identity��
conv1d_20/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_20/conv1d/ExpandDims/dim�
conv1d_20/conv1d/ExpandDims
ExpandDimsinputs(conv1d_20/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d_20/conv1d/ExpandDims�
,conv1d_20/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_20_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02.
,conv1d_20/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_20/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_20/conv1d/ExpandDims_1/dim�
conv1d_20/conv1d/ExpandDims_1
ExpandDims4conv1d_20/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_20/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_20/conv1d/ExpandDims_1�
conv1d_20/conv1dConv2D$conv1d_20/conv1d/ExpandDims:output:0&conv1d_20/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_20/conv1d�
conv1d_20/conv1d/SqueezeSqueezeconv1d_20/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_20/conv1d/Squeeze�
 conv1d_20/BiasAdd/ReadVariableOpReadVariableOp)conv1d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_20/BiasAdd/ReadVariableOp�
conv1d_20/BiasAddBiasAdd!conv1d_20/conv1d/Squeeze:output:0(conv1d_20/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_20/BiasAddz
conv1d_20/ReluReluconv1d_20/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_20/Relu�
conv1d_21/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_21/conv1d/ExpandDims/dim�
conv1d_21/conv1d/ExpandDims
ExpandDimsconv1d_20/Relu:activations:0(conv1d_21/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d_21/conv1d/ExpandDims�
,conv1d_21/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_21_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02.
,conv1d_21/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_21/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_21/conv1d/ExpandDims_1/dim�
conv1d_21/conv1d/ExpandDims_1
ExpandDims4conv1d_21/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_21/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_21/conv1d/ExpandDims_1�
conv1d_21/conv1dConv2D$conv1d_21/conv1d/ExpandDims:output:0&conv1d_21/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_21/conv1d�
conv1d_21/conv1d/SqueezeSqueezeconv1d_21/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_21/conv1d/Squeeze�
 conv1d_21/BiasAdd/ReadVariableOpReadVariableOp)conv1d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_21/BiasAdd/ReadVariableOp�
conv1d_21/BiasAddBiasAdd!conv1d_21/conv1d/Squeeze:output:0(conv1d_21/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_21/BiasAddz
conv1d_21/ReluReluconv1d_21/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_21/Reluy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_10/dropout/Const�
dropout_10/dropout/MulMulconv1d_21/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*+
_output_shapes
:��������� 2
dropout_10/dropout/Mul�
dropout_10/dropout/ShapeShapeconv1d_21/Relu:activations:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape�
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*+
_output_shapes
:��������� *
dtype021
/dropout_10/dropout/random_uniform/RandomUniform�
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_10/dropout/GreaterEqual/y�
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:��������� 2!
dropout_10/dropout/GreaterEqual�
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:��������� 2
dropout_10/dropout/Cast�
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*+
_output_shapes
:��������� 2
dropout_10/dropout/Mul_1�
max_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_10/ExpandDims/dim�
max_pooling1d_10/ExpandDims
ExpandDimsdropout_10/dropout/Mul_1:z:0(max_pooling1d_10/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
max_pooling1d_10/ExpandDims�
max_pooling1d_10/MaxPoolMaxPool$max_pooling1d_10/ExpandDims:output:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling1d_10/MaxPool�
max_pooling1d_10/SqueezeSqueeze!max_pooling1d_10/MaxPool:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims
2
max_pooling1d_10/Squeezeu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_10/Const�
flatten_10/ReshapeReshape!max_pooling1d_10/Squeeze:output:0flatten_10/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_10/Reshape�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMulflatten_10/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_20/BiasAdds
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
dense_20/Relu�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_21/MatMul/ReadVariableOp�
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_21/MatMul�
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_21/BiasAdd/ReadVariableOp�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_21/BiasAdd|
dense_21/SoftmaxSoftmaxdense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_21/Softmaxn
IdentityIdentitydense_21/Softmax:softmax:0*
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
�
I__inference_sequential_10_layer_call_and_return_conditional_losses_517600
conv1d_20_input
conv1d_20_517576
conv1d_20_517578
conv1d_21_517581
conv1d_21_517583
dense_20_517589
dense_20_517591
dense_21_517594
dense_21_517596
identity��!conv1d_20/StatefulPartitionedCall�!conv1d_21/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall�
!conv1d_20/StatefulPartitionedCallStatefulPartitionedCallconv1d_20_inputconv1d_20_517576conv1d_20_517578*
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
E__inference_conv1d_20_layer_call_and_return_conditional_losses_5174252#
!conv1d_20/StatefulPartitionedCall�
!conv1d_21/StatefulPartitionedCallStatefulPartitionedCall*conv1d_20/StatefulPartitionedCall:output:0conv1d_21_517581conv1d_21_517583*
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
E__inference_conv1d_21_layer_call_and_return_conditional_losses_5174572#
!conv1d_21/StatefulPartitionedCall�
dropout_10/PartitionedCallPartitionedCall*conv1d_21/StatefulPartitionedCall:output:0*
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
F__inference_dropout_10_layer_call_and_return_conditional_losses_5174982
dropout_10/PartitionedCall�
 max_pooling1d_10/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
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
L__inference_max_pooling1d_10_layer_call_and_return_conditional_losses_5173992"
 max_pooling1d_10/PartitionedCall�
flatten_10/PartitionedCallPartitionedCall)max_pooling1d_10/PartitionedCall:output:0*
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
F__inference_flatten_10_layer_call_and_return_conditional_losses_5175102
flatten_10/PartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_20_517589dense_20_517591*
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
D__inference_dense_20_layer_call_and_return_conditional_losses_5175292"
 dense_20/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_517594dense_21_517596*
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
D__inference_dense_21_layer_call_and_return_conditional_losses_5175592"
 dense_21/StatefulPartitionedCall�
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0"^conv1d_20/StatefulPartitionedCall"^conv1d_21/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_20/StatefulPartitionedCall!conv1d_20/StatefulPartitionedCall2F
!conv1d_21/StatefulPartitionedCall!conv1d_21/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_20_input
�
�
D__inference_dense_20_layer_call_and_return_conditional_losses_517974

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
�!
�
I__inference_sequential_10_layer_call_and_return_conditional_losses_517573
conv1d_20_input
conv1d_20_517436
conv1d_20_517438
conv1d_21_517468
conv1d_21_517470
dense_20_517540
dense_20_517542
dense_21_517567
dense_21_517569
identity��!conv1d_20/StatefulPartitionedCall�!conv1d_21/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall�"dropout_10/StatefulPartitionedCall�
!conv1d_20/StatefulPartitionedCallStatefulPartitionedCallconv1d_20_inputconv1d_20_517436conv1d_20_517438*
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
E__inference_conv1d_20_layer_call_and_return_conditional_losses_5174252#
!conv1d_20/StatefulPartitionedCall�
!conv1d_21/StatefulPartitionedCallStatefulPartitionedCall*conv1d_20/StatefulPartitionedCall:output:0conv1d_21_517468conv1d_21_517470*
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
E__inference_conv1d_21_layer_call_and_return_conditional_losses_5174572#
!conv1d_21/StatefulPartitionedCall�
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall*conv1d_21/StatefulPartitionedCall:output:0*
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
F__inference_dropout_10_layer_call_and_return_conditional_losses_5174882$
"dropout_10/StatefulPartitionedCall�
 max_pooling1d_10/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling1d_10_layer_call_and_return_conditional_losses_5173992"
 max_pooling1d_10/PartitionedCall�
flatten_10/PartitionedCallPartitionedCall)max_pooling1d_10/PartitionedCall:output:0*
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
F__inference_flatten_10_layer_call_and_return_conditional_losses_5175102
flatten_10/PartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_20_517540dense_20_517542*
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
D__inference_dense_20_layer_call_and_return_conditional_losses_5175292"
 dense_20/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_517567dense_21_517569*
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
D__inference_dense_21_layer_call_and_return_conditional_losses_5175592"
 dense_21/StatefulPartitionedCall�
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0"^conv1d_20/StatefulPartitionedCall"^conv1d_21/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_20/StatefulPartitionedCall!conv1d_20/StatefulPartitionedCall2F
!conv1d_21/StatefulPartitionedCall!conv1d_21/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_20_input
�
�
.__inference_sequential_10_layer_call_fn_517854

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
I__inference_sequential_10_layer_call_and_return_conditional_losses_5176302
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
�
G
+__inference_flatten_10_layer_call_fn_517963

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
F__inference_flatten_10_layer_call_and_return_conditional_losses_5175102
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
�
M
1__inference_max_pooling1d_10_layer_call_fn_517405

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
L__inference_max_pooling1d_10_layer_call_and_return_conditional_losses_5173992
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
�

*__inference_conv1d_21_layer_call_fn_517925

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
E__inference_conv1d_21_layer_call_and_return_conditional_losses_5174572
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
�
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_517947

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
�
�
E__inference_conv1d_21_layer_call_and_return_conditional_losses_517457

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
�
b
F__inference_flatten_10_layer_call_and_return_conditional_losses_517510

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
�!
�
I__inference_sequential_10_layer_call_and_return_conditional_losses_517630

inputs
conv1d_20_517606
conv1d_20_517608
conv1d_21_517611
conv1d_21_517613
dense_20_517619
dense_20_517621
dense_21_517624
dense_21_517626
identity��!conv1d_20/StatefulPartitionedCall�!conv1d_21/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall�"dropout_10/StatefulPartitionedCall�
!conv1d_20/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_20_517606conv1d_20_517608*
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
E__inference_conv1d_20_layer_call_and_return_conditional_losses_5174252#
!conv1d_20/StatefulPartitionedCall�
!conv1d_21/StatefulPartitionedCallStatefulPartitionedCall*conv1d_20/StatefulPartitionedCall:output:0conv1d_21_517611conv1d_21_517613*
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
E__inference_conv1d_21_layer_call_and_return_conditional_losses_5174572#
!conv1d_21/StatefulPartitionedCall�
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall*conv1d_21/StatefulPartitionedCall:output:0*
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
F__inference_dropout_10_layer_call_and_return_conditional_losses_5174882$
"dropout_10/StatefulPartitionedCall�
 max_pooling1d_10/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling1d_10_layer_call_and_return_conditional_losses_5173992"
 max_pooling1d_10/PartitionedCall�
flatten_10/PartitionedCallPartitionedCall)max_pooling1d_10/PartitionedCall:output:0*
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
F__inference_flatten_10_layer_call_and_return_conditional_losses_5175102
flatten_10/PartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_20_517619dense_20_517621*
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
D__inference_dense_20_layer_call_and_return_conditional_losses_5175292"
 dense_20/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_517624dense_21_517626*
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
D__inference_dense_21_layer_call_and_return_conditional_losses_5175592"
 dense_21/StatefulPartitionedCall�
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0"^conv1d_20/StatefulPartitionedCall"^conv1d_21/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_20/StatefulPartitionedCall!conv1d_20/StatefulPartitionedCall2F
!conv1d_21/StatefulPartitionedCall!conv1d_21/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_conv1d_21_layer_call_and_return_conditional_losses_517916

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
�
h
L__inference_max_pooling1d_10_layer_call_and_return_conditional_losses_517399

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
�7
�
I__inference_sequential_10_layer_call_and_return_conditional_losses_517833

inputs9
5conv1d_20_conv1d_expanddims_1_readvariableop_resource-
)conv1d_20_biasadd_readvariableop_resource9
5conv1d_21_conv1d_expanddims_1_readvariableop_resource-
)conv1d_21_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource
identity��
conv1d_20/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_20/conv1d/ExpandDims/dim�
conv1d_20/conv1d/ExpandDims
ExpandDimsinputs(conv1d_20/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d_20/conv1d/ExpandDims�
,conv1d_20/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_20_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02.
,conv1d_20/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_20/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_20/conv1d/ExpandDims_1/dim�
conv1d_20/conv1d/ExpandDims_1
ExpandDims4conv1d_20/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_20/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_20/conv1d/ExpandDims_1�
conv1d_20/conv1dConv2D$conv1d_20/conv1d/ExpandDims:output:0&conv1d_20/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_20/conv1d�
conv1d_20/conv1d/SqueezeSqueezeconv1d_20/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_20/conv1d/Squeeze�
 conv1d_20/BiasAdd/ReadVariableOpReadVariableOp)conv1d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_20/BiasAdd/ReadVariableOp�
conv1d_20/BiasAddBiasAdd!conv1d_20/conv1d/Squeeze:output:0(conv1d_20/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_20/BiasAddz
conv1d_20/ReluReluconv1d_20/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_20/Relu�
conv1d_21/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_21/conv1d/ExpandDims/dim�
conv1d_21/conv1d/ExpandDims
ExpandDimsconv1d_20/Relu:activations:0(conv1d_21/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d_21/conv1d/ExpandDims�
,conv1d_21/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_21_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02.
,conv1d_21/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_21/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_21/conv1d/ExpandDims_1/dim�
conv1d_21/conv1d/ExpandDims_1
ExpandDims4conv1d_21/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_21/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_21/conv1d/ExpandDims_1�
conv1d_21/conv1dConv2D$conv1d_21/conv1d/ExpandDims:output:0&conv1d_21/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_21/conv1d�
conv1d_21/conv1d/SqueezeSqueezeconv1d_21/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_21/conv1d/Squeeze�
 conv1d_21/BiasAdd/ReadVariableOpReadVariableOp)conv1d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_21/BiasAdd/ReadVariableOp�
conv1d_21/BiasAddBiasAdd!conv1d_21/conv1d/Squeeze:output:0(conv1d_21/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_21/BiasAddz
conv1d_21/ReluReluconv1d_21/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_21/Relu�
dropout_10/IdentityIdentityconv1d_21/Relu:activations:0*
T0*+
_output_shapes
:��������� 2
dropout_10/Identity�
max_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_10/ExpandDims/dim�
max_pooling1d_10/ExpandDims
ExpandDimsdropout_10/Identity:output:0(max_pooling1d_10/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
max_pooling1d_10/ExpandDims�
max_pooling1d_10/MaxPoolMaxPool$max_pooling1d_10/ExpandDims:output:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling1d_10/MaxPool�
max_pooling1d_10/SqueezeSqueeze!max_pooling1d_10/MaxPool:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims
2
max_pooling1d_10/Squeezeu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_10/Const�
flatten_10/ReshapeReshape!max_pooling1d_10/Squeeze:output:0flatten_10/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_10/Reshape�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMulflatten_10/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_20/BiasAdds
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
dense_20/Relu�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_21/MatMul/ReadVariableOp�
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_21/MatMul�
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_21/BiasAdd/ReadVariableOp�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_21/BiasAdd|
dense_21/SoftmaxSoftmaxdense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_21/Softmaxn
IdentityIdentitydense_21/Softmax:softmax:0*
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
+__inference_dropout_10_layer_call_fn_517935

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
F__inference_dropout_10_layer_call_and_return_conditional_losses_5174982
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
�
�
I__inference_sequential_10_layer_call_and_return_conditional_losses_517678

inputs
conv1d_20_517654
conv1d_20_517656
conv1d_21_517659
conv1d_21_517661
dense_20_517667
dense_20_517669
dense_21_517672
dense_21_517674
identity��!conv1d_20/StatefulPartitionedCall�!conv1d_21/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall�
!conv1d_20/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_20_517654conv1d_20_517656*
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
E__inference_conv1d_20_layer_call_and_return_conditional_losses_5174252#
!conv1d_20/StatefulPartitionedCall�
!conv1d_21/StatefulPartitionedCallStatefulPartitionedCall*conv1d_20/StatefulPartitionedCall:output:0conv1d_21_517659conv1d_21_517661*
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
E__inference_conv1d_21_layer_call_and_return_conditional_losses_5174572#
!conv1d_21/StatefulPartitionedCall�
dropout_10/PartitionedCallPartitionedCall*conv1d_21/StatefulPartitionedCall:output:0*
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
F__inference_dropout_10_layer_call_and_return_conditional_losses_5174982
dropout_10/PartitionedCall�
 max_pooling1d_10/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
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
L__inference_max_pooling1d_10_layer_call_and_return_conditional_losses_5173992"
 max_pooling1d_10/PartitionedCall�
flatten_10/PartitionedCallPartitionedCall)max_pooling1d_10/PartitionedCall:output:0*
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
F__inference_flatten_10_layer_call_and_return_conditional_losses_5175102
flatten_10/PartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_20_517667dense_20_517669*
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
D__inference_dense_20_layer_call_and_return_conditional_losses_5175292"
 dense_20/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_517672dense_21_517674*
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
D__inference_dense_21_layer_call_and_return_conditional_losses_5175592"
 dense_21/StatefulPartitionedCall�
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0"^conv1d_20/StatefulPartitionedCall"^conv1d_21/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_20/StatefulPartitionedCall!conv1d_20/StatefulPartitionedCall2F
!conv1d_21/StatefulPartitionedCall!conv1d_21/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
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
conv1d_20_input<
!serving_default_conv1d_20_input:0���������<
dense_210
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
_tf_keras_sequential�6{"training_config": {"weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"beta_1": 0.8999999761581421, "epsilon": 1e-07, "beta_2": 0.9990000128746033, "learning_rate": 9.999999747378752e-05, "decay": 0.0, "name": "Adam", "amsgrad": false}}, "loss": "categorical_crossentropy", "metrics": ["accuracy"], "loss_weights": null}, "expects_training_arg": true, "model_config": {"class_name": "Sequential", "config": {"layers": [{"class_name": "InputLayer", "config": {"sparse": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "dtype": "float32", "ragged": false, "name": "conv1d_20_input"}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "name": "conv1d_20", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "name": "conv1d_21", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Dropout", "config": {"dtype": "float32", "seed": null, "rate": 0.5, "noise_shape": null, "name": "dropout_10", "trainable": true}}, {"class_name": "MaxPooling1D", "config": {"strides": {"class_name": "__tuple__", "items": [2]}, "name": "max_pooling1d_10", "data_format": "channels_last", "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "trainable": true, "dtype": "float32"}}, {"class_name": "Flatten", "config": {"dtype": "float32", "name": "flatten_10", "data_format": "channels_last", "trainable": true}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 100, "kernel_constraint": null, "activity_regularizer": null, "activation": "relu", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_20", "trainable": true, "kernel_regularizer": null}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 2, "kernel_constraint": null, "activity_regularizer": null, "activation": "softmax", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_21", "trainable": true, "kernel_regularizer": null}}], "name": "sequential_10"}}, "class_name": "Sequential", "must_restore_from_config": false, "dtype": "float32", "config": {"layers": [{"class_name": "InputLayer", "config": {"sparse": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "dtype": "float32", "ragged": false, "name": "conv1d_20_input"}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "name": "conv1d_20", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "name": "conv1d_21", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Dropout", "config": {"dtype": "float32", "seed": null, "rate": 0.5, "noise_shape": null, "name": "dropout_10", "trainable": true}}, {"class_name": "MaxPooling1D", "config": {"strides": {"class_name": "__tuple__", "items": [2]}, "name": "max_pooling1d_10", "data_format": "channels_last", "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "trainable": true, "dtype": "float32"}}, {"class_name": "Flatten", "config": {"dtype": "float32", "name": "flatten_10", "data_format": "channels_last", "trainable": true}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 100, "kernel_constraint": null, "activity_regularizer": null, "activation": "relu", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_20", "trainable": true, "kernel_regularizer": null}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 2, "kernel_constraint": null, "activity_regularizer": null, "activation": "softmax", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_21", "trainable": true, "kernel_regularizer": null}}], "name": "sequential_10"}, "keras_version": "2.4.0", "batch_input_shape": null, "is_graph_network": true, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 5}, "max_ndim": null, "min_ndim": 3}}, "name": "sequential_10", "backend": "tensorflow", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 5]}}
�


kernel
bias
	keras_api
regularization_losses
	variables
trainable_variables
}__call__
*~&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"config": {"dtype": "float32", "use_bias": true, "strides": {"class_name": "__tuple__", "items": [1]}, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "activity_regularizer": null, "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "groups": 1, "filters": 32, "bias_regularizer": null, "activation": "relu", "name": "conv1d_20", "kernel_constraint": null, "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "name": "conv1d_20", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 5}, "max_ndim": null, "min_ndim": 3}}, "class_name": "Conv1D", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 5]}, "trainable": true}
�	

kernel
bias
	keras_api
regularization_losses
	variables
trainable_variables
__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"config": {"dtype": "float32", "use_bias": true, "strides": {"class_name": "__tuple__", "items": [1]}, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "activity_regularizer": null, "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "groups": 1, "filters": 32, "bias_regularizer": null, "activation": "relu", "name": "conv1d_21", "kernel_constraint": null, "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": null, "name": "conv1d_21", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 32}, "max_ndim": null, "min_ndim": 3}}, "class_name": "Conv1D", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 32]}, "trainable": true}
�
	keras_api
regularization_losses
	variables
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"must_restore_from_config": false, "batch_input_shape": null, "name": "dropout_10", "config": {"trainable": true, "seed": null, "rate": 0.5, "noise_shape": null, "name": "dropout_10", "dtype": "float32"}, "class_name": "Dropout", "expects_training_arg": true, "dtype": "float32", "stateful": false, "trainable": true}
�
	keras_api
regularization_losses
 	variables
!trainable_variables
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"config": {"trainable": true, "strides": {"class_name": "__tuple__", "items": [2]}, "data_format": "channels_last", "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "dtype": "float32", "name": "max_pooling1d_10"}, "must_restore_from_config": false, "batch_input_shape": null, "name": "max_pooling1d_10", "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "axes": {}, "max_ndim": null, "min_ndim": null}}, "class_name": "MaxPooling1D", "expects_training_arg": false, "dtype": "float32", "stateful": false, "trainable": true}
�
"	keras_api
#regularization_losses
$	variables
%trainable_variables
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"config": {"name": "flatten_10", "trainable": true, "data_format": "channels_last", "dtype": "float32"}, "must_restore_from_config": false, "batch_input_shape": null, "name": "flatten_10", "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {}, "max_ndim": null, "min_ndim": 1}}, "class_name": "Flatten", "expects_training_arg": false, "dtype": "float32", "stateful": false, "trainable": true}
�

&kernel
'bias
(	keras_api
)regularization_losses
*	variables
+trainable_variables
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"config": {"dtype": "float32", "use_bias": true, "units": 100, "kernel_constraint": null, "activity_regularizer": null, "activation": "relu", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_20", "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": null, "name": "dense_20", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 192}, "max_ndim": null, "min_ndim": 2}}, "class_name": "Dense", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 192]}, "trainable": true}
�

,kernel
-bias
.	keras_api
/regularization_losses
0	variables
1trainable_variables
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"config": {"dtype": "float32", "use_bias": true, "units": 2, "kernel_constraint": null, "activity_regularizer": null, "activation": "softmax", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_21", "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": null, "name": "dense_21", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 100}, "max_ndim": null, "min_ndim": 2}}, "class_name": "Dense", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "trainable": true}
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
&:$ 2conv1d_20/kernel
: 2conv1d_20/bias
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
&:$  2conv1d_21/kernel
: 2conv1d_21/bias
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
": 	�d2dense_20/kernel
:d2dense_20/bias
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
!:d2dense_21/kernel
:2dense_21/bias
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
+:) 2Adam/conv1d_20/kernel/m
!: 2Adam/conv1d_20/bias/m
+:)  2Adam/conv1d_21/kernel/m
!: 2Adam/conv1d_21/bias/m
':%	�d2Adam/dense_20/kernel/m
 :d2Adam/dense_20/bias/m
&:$d2Adam/dense_21/kernel/m
 :2Adam/dense_21/bias/m
+:) 2Adam/conv1d_20/kernel/v
!: 2Adam/conv1d_20/bias/v
+:)  2Adam/conv1d_21/kernel/v
!: 2Adam/conv1d_21/bias/v
':%	�d2Adam/dense_20/kernel/v
 :d2Adam/dense_20/bias/v
&:$d2Adam/dense_21/kernel/v
 :2Adam/dense_21/bias/v
�2�
!__inference__wrapped_model_517390�
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
conv1d_20_input���������
�2�
.__inference_sequential_10_layer_call_fn_517649
.__inference_sequential_10_layer_call_fn_517875
.__inference_sequential_10_layer_call_fn_517854
.__inference_sequential_10_layer_call_fn_517697�
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
I__inference_sequential_10_layer_call_and_return_conditional_losses_517600
I__inference_sequential_10_layer_call_and_return_conditional_losses_517833
I__inference_sequential_10_layer_call_and_return_conditional_losses_517784
I__inference_sequential_10_layer_call_and_return_conditional_losses_517573�
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
*__inference_conv1d_20_layer_call_fn_517900�
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
E__inference_conv1d_20_layer_call_and_return_conditional_losses_517891�
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
*__inference_conv1d_21_layer_call_fn_517925�
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
E__inference_conv1d_21_layer_call_and_return_conditional_losses_517916�
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
+__inference_dropout_10_layer_call_fn_517935
+__inference_dropout_10_layer_call_fn_517930�
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
F__inference_dropout_10_layer_call_and_return_conditional_losses_517947
F__inference_dropout_10_layer_call_and_return_conditional_losses_517952�
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
1__inference_max_pooling1d_10_layer_call_fn_517405�
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
L__inference_max_pooling1d_10_layer_call_and_return_conditional_losses_517399�
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
+__inference_flatten_10_layer_call_fn_517963�
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
F__inference_flatten_10_layer_call_and_return_conditional_losses_517958�
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
)__inference_dense_20_layer_call_fn_517983�
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
D__inference_dense_20_layer_call_and_return_conditional_losses_517974�
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
)__inference_dense_21_layer_call_fn_517992�
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
D__inference_dense_21_layer_call_and_return_conditional_losses_518003�
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
$__inference_signature_wrapper_517728conv1d_20_input�
!__inference__wrapped_model_517390}&',-<�9
2�/
-�*
conv1d_20_input���������
� "3�0
.
dense_21"�
dense_21����������
E__inference_conv1d_20_layer_call_and_return_conditional_losses_517891d3�0
)�&
$�!
inputs���������
� ")�&
�
0��������� 
� �
*__inference_conv1d_20_layer_call_fn_517900W3�0
)�&
$�!
inputs���������
� "���������� �
E__inference_conv1d_21_layer_call_and_return_conditional_losses_517916d3�0
)�&
$�!
inputs��������� 
� ")�&
�
0��������� 
� �
*__inference_conv1d_21_layer_call_fn_517925W3�0
)�&
$�!
inputs��������� 
� "���������� �
D__inference_dense_20_layer_call_and_return_conditional_losses_517974]&'0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� }
)__inference_dense_20_layer_call_fn_517983P&'0�-
&�#
!�
inputs����������
� "����������d�
D__inference_dense_21_layer_call_and_return_conditional_losses_518003\,-/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� |
)__inference_dense_21_layer_call_fn_517992O,-/�,
%�"
 �
inputs���������d
� "�����������
F__inference_dropout_10_layer_call_and_return_conditional_losses_517947d7�4
-�*
$�!
inputs��������� 
p
� ")�&
�
0��������� 
� �
F__inference_dropout_10_layer_call_and_return_conditional_losses_517952d7�4
-�*
$�!
inputs��������� 
p 
� ")�&
�
0��������� 
� �
+__inference_dropout_10_layer_call_fn_517930W7�4
-�*
$�!
inputs��������� 
p
� "���������� �
+__inference_dropout_10_layer_call_fn_517935W7�4
-�*
$�!
inputs��������� 
p 
� "���������� �
F__inference_flatten_10_layer_call_and_return_conditional_losses_517958]3�0
)�&
$�!
inputs��������� 
� "&�#
�
0����������
� 
+__inference_flatten_10_layer_call_fn_517963P3�0
)�&
$�!
inputs��������� 
� "������������
L__inference_max_pooling1d_10_layer_call_and_return_conditional_losses_517399�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
1__inference_max_pooling1d_10_layer_call_fn_517405wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
I__inference_sequential_10_layer_call_and_return_conditional_losses_517573w&',-D�A
:�7
-�*
conv1d_20_input���������
p

 
� "%�"
�
0���������
� �
I__inference_sequential_10_layer_call_and_return_conditional_losses_517600w&',-D�A
:�7
-�*
conv1d_20_input���������
p 

 
� "%�"
�
0���������
� �
I__inference_sequential_10_layer_call_and_return_conditional_losses_517784n&',-;�8
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
I__inference_sequential_10_layer_call_and_return_conditional_losses_517833n&',-;�8
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
.__inference_sequential_10_layer_call_fn_517649j&',-D�A
:�7
-�*
conv1d_20_input���������
p

 
� "�����������
.__inference_sequential_10_layer_call_fn_517697j&',-D�A
:�7
-�*
conv1d_20_input���������
p 

 
� "�����������
.__inference_sequential_10_layer_call_fn_517854a&',-;�8
1�.
$�!
inputs���������
p

 
� "�����������
.__inference_sequential_10_layer_call_fn_517875a&',-;�8
1�.
$�!
inputs���������
p 

 
� "�����������
$__inference_signature_wrapper_517728�&',-O�L
� 
E�B
@
conv1d_20_input-�*
conv1d_20_input���������"3�0
.
dense_21"�
dense_21���������