Шє
╤г
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
dtypetypeИ
╛
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878оГ
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
: *
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
: *
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:  *
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
: *
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└d*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	└d*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:d*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:d*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
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
М
Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_2/kernel/m
Е
*Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/m*"
_output_shapes
: *
dtype0
А
Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_2/bias/m
y
(Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/m*
_output_shapes
: *
dtype0
М
Adam/conv1d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv1d_3/kernel/m
Е
*Adam/conv1d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/m*"
_output_shapes
:  *
dtype0
А
Adam/conv1d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_3/bias/m
y
(Adam/conv1d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/m*
_output_shapes
: *
dtype0
З
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└d*&
shared_nameAdam/dense_2/kernel/m
А
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	└d*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:d*
dtype0
Ж
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
М
Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_2/kernel/v
Е
*Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/v*"
_output_shapes
: *
dtype0
А
Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_2/bias/v
y
(Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/v*
_output_shapes
: *
dtype0
М
Adam/conv1d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv1d_3/kernel/v
Е
*Adam/conv1d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/v*"
_output_shapes
:  *
dtype0
А
Adam/conv1d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_3/bias/v
y
(Adam/conv1d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/v*
_output_shapes
: *
dtype0
З
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└d*&
shared_nameAdam/dense_2/kernel/v
А
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	└d*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:d*
dtype0
Ж
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ъ4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*е4
valueЫ4BШ4 BС4
┤
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
╨
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy
н

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
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
н
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
[Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
н
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
н
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
н
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
н
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
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
н
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
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
н
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
~|
VARIABLE_VALUEAdam/conv1d_2/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_2/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_2/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_2/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Й
serving_default_conv1d_2_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
─
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_2_inputconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_93621
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ш
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv1d_2/kernel/m/Read/ReadVariableOp(Adam/conv1d_2/bias/m/Read/ReadVariableOp*Adam/conv1d_3/kernel/m/Read/ReadVariableOp(Adam/conv1d_3/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp*Adam/conv1d_2/kernel/v/Read/ReadVariableOp(Adam/conv1d_2/bias/v/Read/ReadVariableOp*Adam/conv1d_3/kernel/v/Read/ReadVariableOp(Adam/conv1d_3/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst*.
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
GPU 2J 8В *'
f"R 
__inference__traced_save_94018
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv1d_2/kernel/mAdam/conv1d_2/bias/mAdam/conv1d_3/kernel/mAdam/conv1d_3/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/conv1d_2/kernel/vAdam/conv1d_2/bias/vAdam/conv1d_3/kernel/vAdam/conv1d_3/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*-
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_94127Ю°
ъ
}
(__inference_conv1d_3_layer_call_fn_93802

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_933532
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:          2

Identity"
identityIdentity:output:0*2
_input_shapes!
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
п
к
B__inference_dense_3_layer_call_and_return_conditional_losses_93887

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d:::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
■6
ш
G__inference_sequential_1_layer_call_and_return_conditional_losses_93768

inputs8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityИЛ
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_2/conv1d/ExpandDims/dim▒
conv1d_2/conv1d/ExpandDims
ExpandDimsinputs'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2
conv1d_2/conv1d/ExpandDims╙
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim█
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_2/conv1d/ExpandDims_1█
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv1d_2/conv1dн
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d_2/conv1d/Squeezeз
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_2/BiasAdd/ReadVariableOp░
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:          2
conv1d_2/ReluЛ
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_3/conv1d/ExpandDims/dim╞
conv1d_3/conv1d/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2
conv1d_3/conv1d/ExpandDims╙
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim█
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_3/conv1d/ExpandDims_1█
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv1d_3/conv1dн
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d_3/conv1d/Squeezeз
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_3/BiasAdd/ReadVariableOp░
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:          2
conv1d_3/ReluЗ
dropout_1/IdentityIdentityconv1d_3/Relu:activations:0*
T0*+
_output_shapes
:          2
dropout_1/IdentityВ
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim╞
max_pooling1d_1/ExpandDims
ExpandDimsdropout_1/Identity:output:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2
max_pooling1d_1/ExpandDims╧
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPoolм
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:          *
squeeze_dims
2
max_pooling1d_1/Squeezes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    └   2
flatten_1/Constа
flatten_1/ReshapeReshape max_pooling1d_1/Squeeze:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         └2
flatten_1/Reshapeж
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	└d*
dtype02
dense_2/MatMul/ReadVariableOpЯ
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
dense_2/Reluе
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_3/MatMul/ReadVariableOpЯ
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_3/Softmaxm
IdentityIdentitydense_3/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         :::::::::S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╕@
ш
G__inference_sequential_1_layer_call_and_return_conditional_losses_93719

inputs8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityИЛ
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_2/conv1d/ExpandDims/dim▒
conv1d_2/conv1d/ExpandDims
ExpandDimsinputs'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2
conv1d_2/conv1d/ExpandDims╙
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim█
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_2/conv1d/ExpandDims_1█
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv1d_2/conv1dн
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d_2/conv1d/Squeezeз
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_2/BiasAdd/ReadVariableOp░
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:          2
conv1d_2/ReluЛ
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_3/conv1d/ExpandDims/dim╞
conv1d_3/conv1d/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2
conv1d_3/conv1d/ExpandDims╙
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim█
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_3/conv1d/ExpandDims_1█
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv1d_3/conv1dн
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d_3/conv1d/Squeezeз
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_3/BiasAdd/ReadVariableOp░
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:          2
conv1d_3/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Constк
dropout_1/dropout/MulMulconv1d_3/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:          2
dropout_1/dropout/Mul}
dropout_1/dropout/ShapeShapeconv1d_3/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape╓
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:          *
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yъ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:          2 
dropout_1/dropout/GreaterEqualб
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:          2
dropout_1/dropout/Castж
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:          2
dropout_1/dropout/Mul_1В
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim╞
max_pooling1d_1/ExpandDims
ExpandDimsdropout_1/dropout/Mul_1:z:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2
max_pooling1d_1/ExpandDims╧
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPoolм
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:          *
squeeze_dims
2
max_pooling1d_1/Squeezes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    └   2
flatten_1/Constа
flatten_1/ReshapeReshape max_pooling1d_1/Squeeze:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         └2
flatten_1/Reshapeж
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	└d*
dtype02
dense_2/MatMul/ReadVariableOpЯ
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
dense_2/Reluе
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_3/MatMul/ReadVariableOpЯ
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_3/Softmaxm
IdentityIdentitydense_3/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         :::::::::S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Щ
╕
C__inference_conv1d_2_layer_call_and_return_conditional_losses_93784

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:          2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:          2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :::S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ЗH
│
__inference__traced_save_94018
file_prefix.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_3_kernel_m_read_readvariableop3
/savev2_adam_conv1d_3_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop5
1savev2_adam_conv1d_3_kernel_v_read_readvariableop3
/savev2_adam_conv1d_3_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a95e95d6ba9f443a858041b9e89f071c/part2	
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╞
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*╪
value╬B╦"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╠
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЯ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop1savev2_adam_conv1d_3_kernel_m_read_readvariableop/savev2_adam_conv1d_3_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop1savev2_adam_conv1d_3_kernel_v_read_readvariableop/savev2_adam_conv1d_3_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*Ж
_input_shapesЇ
ё: : : :  : :	└d:d:d:: : : : : : : : : : : :  : :	└d:d:d:: : :  : :	└d:d:d:: 2(
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
:	└d: 
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
:	└d: 
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
:	└d: 
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
У
┌
#__inference_signature_wrapper_93621
conv1d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallconv1d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_932832
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_2_input
╥ 
░
G__inference_sequential_1_layer_call_and_return_conditional_losses_93523

inputs
conv1d_2_93499
conv1d_2_93501
conv1d_3_93504
conv1d_3_93506
dense_2_93512
dense_2_93514
dense_3_93517
dense_3_93519
identityИв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallХ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_2_93499conv1d_2_93501*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_933182"
 conv1d_2/StatefulPartitionedCall╕
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0conv1d_3_93504conv1d_3_93506*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_933532"
 conv1d_3/StatefulPartitionedCallХ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_933812#
!dropout_1/StatefulPartitionedCallР
max_pooling1d_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_932952!
max_pooling1d_1/PartitionedCall∙
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_934062
flatten_1/PartitionedCallи
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_93512dense_2_93514*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_934252!
dense_2/StatefulPartitionedCallо
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_93517dense_3_93519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_934492!
dense_3/StatefulPartitionedCallк
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         ::::::::2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ч
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_93295

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           2

ExpandDims▒
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
┼Л
ч
!__inference__traced_restore_94127
file_prefix$
 assignvariableop_conv1d_2_kernel$
 assignvariableop_1_conv1d_2_bias&
"assignvariableop_2_conv1d_3_kernel$
 assignvariableop_3_conv1d_3_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1.
*assignvariableop_17_adam_conv1d_2_kernel_m,
(assignvariableop_18_adam_conv1d_2_bias_m.
*assignvariableop_19_adam_conv1d_3_kernel_m,
(assignvariableop_20_adam_conv1d_3_bias_m-
)assignvariableop_21_adam_dense_2_kernel_m+
'assignvariableop_22_adam_dense_2_bias_m-
)assignvariableop_23_adam_dense_3_kernel_m+
'assignvariableop_24_adam_dense_3_bias_m.
*assignvariableop_25_adam_conv1d_2_kernel_v,
(assignvariableop_26_adam_conv1d_2_bias_v.
*assignvariableop_27_adam_conv1d_3_kernel_v,
(assignvariableop_28_adam_conv1d_3_bias_v-
)assignvariableop_29_adam_dense_2_kernel_v+
'assignvariableop_30_adam_dense_2_bias_v-
)assignvariableop_31_adam_dense_3_kernel_v+
'assignvariableop_32_adam_dense_3_bias_v
identity_34ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╠
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*╪
value╬B╦"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╥
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╪
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_conv1d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2з
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3е
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ж
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5д
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ж
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7д
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8б
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9г
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10з
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ж
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12о
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13б
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14б
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15г
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16г
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17▓
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv1d_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18░
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv1d_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19▓
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv1d_3_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20░
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv1d_3_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21▒
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22п
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23▒
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_3_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24п
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_3_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▓
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv1d_2_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv1d_2_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27▓
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv1d_3_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28░
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv1d_3_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29▒
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30п
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31▒
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32п
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp┤
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33з
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*Ы
_input_shapesЙ
Ж: :::::::::::::::::::::::::::::::::2$
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
Щ
╕
C__inference_conv1d_3_layer_call_and_return_conditional_losses_93353

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:          2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:          2

Identity"
identityIdentity:output:0*2
_input_shapes!
:          :::S O
+
_output_shapes
:          
 
_user_specified_nameinputs
ъ
}
(__inference_conv1d_2_layer_call_fn_93793

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_933182
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:          2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Щ
╕
C__inference_conv1d_2_layer_call_and_return_conditional_losses_93318

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:          2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:          2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :::S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╫
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_93391

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:          2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
┌
|
'__inference_dense_2_layer_call_fn_93865

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_934252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
о
b
)__inference_dropout_1_layer_call_fn_93823

inputs
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_933812
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:          2

Identity"
identityIdentity:output:0**
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
╫
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_93845

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:          2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
чC
▒
 __inference__wrapped_model_93283
conv1d_2_inputE
Asequential_1_conv1d_2_conv1d_expanddims_1_readvariableop_resource9
5sequential_1_conv1d_2_biasadd_readvariableop_resourceE
Asequential_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource9
5sequential_1_conv1d_3_biasadd_readvariableop_resource7
3sequential_1_dense_2_matmul_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource7
3sequential_1_dense_3_matmul_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource
identityИе
+sequential_1/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+sequential_1/conv1d_2/conv1d/ExpandDims/dimр
'sequential_1/conv1d_2/conv1d/ExpandDims
ExpandDimsconv1d_2_input4sequential_1/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2)
'sequential_1/conv1d_2/conv1d/ExpandDims·
8sequential_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_1_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02:
8sequential_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpа
-sequential_1/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_1/conv1d_2/conv1d/ExpandDims_1/dimП
)sequential_1/conv1d_2/conv1d/ExpandDims_1
ExpandDims@sequential_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_1/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2+
)sequential_1/conv1d_2/conv1d/ExpandDims_1П
sequential_1/conv1d_2/conv1dConv2D0sequential_1/conv1d_2/conv1d/ExpandDims:output:02sequential_1/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
sequential_1/conv1d_2/conv1d╘
$sequential_1/conv1d_2/conv1d/SqueezeSqueeze%sequential_1/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2&
$sequential_1/conv1d_2/conv1d/Squeeze╬
,sequential_1/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv1d_2/BiasAdd/ReadVariableOpф
sequential_1/conv1d_2/BiasAddBiasAdd-sequential_1/conv1d_2/conv1d/Squeeze:output:04sequential_1/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2
sequential_1/conv1d_2/BiasAddЮ
sequential_1/conv1d_2/ReluRelu&sequential_1/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:          2
sequential_1/conv1d_2/Reluе
+sequential_1/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+sequential_1/conv1d_3/conv1d/ExpandDims/dim·
'sequential_1/conv1d_3/conv1d/ExpandDims
ExpandDims(sequential_1/conv1d_2/Relu:activations:04sequential_1/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2)
'sequential_1/conv1d_3/conv1d/ExpandDims·
8sequential_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02:
8sequential_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpа
-sequential_1/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_1/conv1d_3/conv1d/ExpandDims_1/dimП
)sequential_1/conv1d_3/conv1d/ExpandDims_1
ExpandDims@sequential_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_1/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2+
)sequential_1/conv1d_3/conv1d/ExpandDims_1П
sequential_1/conv1d_3/conv1dConv2D0sequential_1/conv1d_3/conv1d/ExpandDims:output:02sequential_1/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
sequential_1/conv1d_3/conv1d╘
$sequential_1/conv1d_3/conv1d/SqueezeSqueeze%sequential_1/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2&
$sequential_1/conv1d_3/conv1d/Squeeze╬
,sequential_1/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv1d_3/BiasAdd/ReadVariableOpф
sequential_1/conv1d_3/BiasAddBiasAdd-sequential_1/conv1d_3/conv1d/Squeeze:output:04sequential_1/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2
sequential_1/conv1d_3/BiasAddЮ
sequential_1/conv1d_3/ReluRelu&sequential_1/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:          2
sequential_1/conv1d_3/Reluо
sequential_1/dropout_1/IdentityIdentity(sequential_1/conv1d_3/Relu:activations:0*
T0*+
_output_shapes
:          2!
sequential_1/dropout_1/IdentityЬ
+sequential_1/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential_1/max_pooling1d_1/ExpandDims/dim·
'sequential_1/max_pooling1d_1/ExpandDims
ExpandDims(sequential_1/dropout_1/Identity:output:04sequential_1/max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2)
'sequential_1/max_pooling1d_1/ExpandDimsЎ
$sequential_1/max_pooling1d_1/MaxPoolMaxPool0sequential_1/max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling1d_1/MaxPool╙
$sequential_1/max_pooling1d_1/SqueezeSqueeze-sequential_1/max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:          *
squeeze_dims
2&
$sequential_1/max_pooling1d_1/SqueezeН
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    └   2
sequential_1/flatten_1/Const╘
sequential_1/flatten_1/ReshapeReshape-sequential_1/max_pooling1d_1/Squeeze:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:         └2 
sequential_1/flatten_1/Reshape═
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	└d*
dtype02,
*sequential_1/dense_2/MatMul/ReadVariableOp╙
sequential_1/dense_2/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
sequential_1/dense_2/MatMul╦
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp╒
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
sequential_1/dense_2/BiasAddЧ
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
sequential_1/dense_2/Relu╠
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOp╙
sequential_1/dense_3/MatMulMatMul'sequential_1/dense_2/Relu:activations:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_3/MatMul╦
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp╒
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_3/BiasAddа
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential_1/dense_3/Softmaxz
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         :::::::::[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_2_input
к
к
B__inference_dense_2_layer_call_and_return_conditional_losses_93876

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └:::P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
б
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_93840

inputs
identityИc
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
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:          2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:          2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:          2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:          2

Identity"
identityIdentity:output:0**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
л
█
,__inference_sequential_1_layer_call_fn_93642

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_935232
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╪
|
'__inference_dense_3_layer_call_fn_93896

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_934492
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
л
█
,__inference_sequential_1_layer_call_fn_93663

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_935712
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
├
у
,__inference_sequential_1_layer_call_fn_93542
conv1d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityИвStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallconv1d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_935232
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_2_input
║
Ф
G__inference_sequential_1_layer_call_and_return_conditional_losses_93493
conv1d_2_input
conv1d_2_93469
conv1d_2_93471
conv1d_3_93474
conv1d_3_93476
dense_2_93482
dense_2_93484
dense_3_93487
dense_3_93489
identityИв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallЭ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallconv1d_2_inputconv1d_2_93469conv1d_2_93471*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_933182"
 conv1d_2/StatefulPartitionedCall╕
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0conv1d_3_93474conv1d_3_93476*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_933532"
 conv1d_3/StatefulPartitionedCall¤
dropout_1/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_933912
dropout_1/PartitionedCallИ
max_pooling1d_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_932952!
max_pooling1d_1/PartitionedCall∙
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_934062
flatten_1/PartitionedCallи
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_93482dense_2_93484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_934252!
dense_2/StatefulPartitionedCallо
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_93487dense_3_93489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_934492!
dense_3/StatefulPartitionedCallЖ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         ::::::::2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_2_input
б
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_93381

inputs
identityИc
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
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:          2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:          2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:          2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:          2

Identity"
identityIdentity:output:0**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
├
у
,__inference_sequential_1_layer_call_fn_93590
conv1d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityИвStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallconv1d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_935712
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_2_input
п
к
B__inference_dense_3_layer_call_and_return_conditional_losses_93449

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d:::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Щ
╕
C__inference_conv1d_3_layer_call_and_return_conditional_losses_93818

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:          2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:          2

Identity"
identityIdentity:output:0*2
_input_shapes!
:          :::S O
+
_output_shapes
:          
 
_user_specified_nameinputs
ў
K
/__inference_max_pooling1d_1_layer_call_fn_93298

inputs
identity▐
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_932952
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
в
М
G__inference_sequential_1_layer_call_and_return_conditional_losses_93571

inputs
conv1d_2_93547
conv1d_2_93549
conv1d_3_93552
conv1d_3_93554
dense_2_93560
dense_2_93562
dense_3_93565
dense_3_93567
identityИв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallХ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_2_93547conv1d_2_93549*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_933182"
 conv1d_2/StatefulPartitionedCall╕
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0conv1d_3_93552conv1d_3_93554*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_933532"
 conv1d_3/StatefulPartitionedCall¤
dropout_1/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_933912
dropout_1/PartitionedCallИ
max_pooling1d_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_932952!
max_pooling1d_1/PartitionedCall∙
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_934062
flatten_1/PartitionedCallи
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_93560dense_2_93562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_934252!
dense_2/StatefulPartitionedCallо
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_93565dense_3_93567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_934492!
dense_3/StatefulPartitionedCallЖ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         ::::::::2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
в
E
)__inference_dropout_1_layer_call_fn_93828

inputs
identity╞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_933912
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:          2

Identity"
identityIdentity:output:0**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
┤
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_93856

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    └   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
к
к
B__inference_dense_2_layer_call_and_return_conditional_losses_93425

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └:::P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
ъ 
╕
G__inference_sequential_1_layer_call_and_return_conditional_losses_93466
conv1d_2_input
conv1d_2_93329
conv1d_2_93331
conv1d_3_93361
conv1d_3_93363
dense_2_93433
dense_2_93435
dense_3_93460
dense_3_93462
identityИв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallЭ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallconv1d_2_inputconv1d_2_93329conv1d_2_93331*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_933182"
 conv1d_2/StatefulPartitionedCall╕
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0conv1d_3_93361conv1d_3_93363*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_933532"
 conv1d_3/StatefulPartitionedCallХ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_933812#
!dropout_1/StatefulPartitionedCallР
max_pooling1d_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_932952!
max_pooling1d_1/PartitionedCall∙
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_934062
flatten_1/PartitionedCallи
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_93433dense_2_93435*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_934252!
dense_2/StatefulPartitionedCallо
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_93460dense_3_93462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_934492!
dense_3/StatefulPartitionedCallк
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:         ::::::::2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_nameconv1d_2_input
┤
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_93406

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    └   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
Ь
E
)__inference_flatten_1_layer_call_fn_93850

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_934062
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╝
serving_defaultи
M
conv1d_2_input;
 serving_default_conv1d_2_input:0         ;
dense_30
StatefulPartitionedCall:0         tensorflow/serving/predict:аш
Ї9
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
*|&call_and_return_all_conditional_losses"ц6
_tf_keras_sequential╟6{"training_config": {"weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"beta_1": 0.8999999761581421, "epsilon": 1e-07, "beta_2": 0.9990000128746033, "learning_rate": 9.999999747378752e-05, "decay": 0.0, "name": "Adam", "amsgrad": false}}, "loss": "categorical_crossentropy", "metrics": ["accuracy"], "loss_weights": null}, "expects_training_arg": true, "model_config": {"class_name": "Sequential", "config": {"layers": [{"class_name": "InputLayer", "config": {"sparse": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "dtype": "float32", "ragged": false, "name": "conv1d_2_input"}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "name": "conv1d_2", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "name": "conv1d_3", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Dropout", "config": {"dtype": "float32", "seed": null, "rate": 0.5, "noise_shape": null, "name": "dropout_1", "trainable": true}}, {"class_name": "MaxPooling1D", "config": {"strides": {"class_name": "__tuple__", "items": [2]}, "name": "max_pooling1d_1", "data_format": "channels_last", "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "trainable": true, "dtype": "float32"}}, {"class_name": "Flatten", "config": {"dtype": "float32", "name": "flatten_1", "data_format": "channels_last", "trainable": true}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 100, "kernel_constraint": null, "activity_regularizer": null, "activation": "relu", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_2", "trainable": true, "kernel_regularizer": null}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 2, "kernel_constraint": null, "activity_regularizer": null, "activation": "softmax", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_3", "trainable": true, "kernel_regularizer": null}}], "name": "sequential_1"}}, "class_name": "Sequential", "must_restore_from_config": false, "dtype": "float32", "config": {"layers": [{"class_name": "InputLayer", "config": {"sparse": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "dtype": "float32", "ragged": false, "name": "conv1d_2_input"}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "name": "conv1d_2", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Conv1D", "config": {"use_bias": true, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activity_regularizer": null, "padding": "valid", "dtype": "float32", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "name": "conv1d_3", "groups": 1, "filters": 32, "activation": "relu", "bias_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1]}, "trainable": true, "kernel_constraint": null, "kernel_regularizer": null}}, {"class_name": "Dropout", "config": {"dtype": "float32", "seed": null, "rate": 0.5, "noise_shape": null, "name": "dropout_1", "trainable": true}}, {"class_name": "MaxPooling1D", "config": {"strides": {"class_name": "__tuple__", "items": [2]}, "name": "max_pooling1d_1", "data_format": "channels_last", "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "trainable": true, "dtype": "float32"}}, {"class_name": "Flatten", "config": {"dtype": "float32", "name": "flatten_1", "data_format": "channels_last", "trainable": true}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 100, "kernel_constraint": null, "activity_regularizer": null, "activation": "relu", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_2", "trainable": true, "kernel_regularizer": null}}, {"class_name": "Dense", "config": {"dtype": "float32", "use_bias": true, "units": 2, "kernel_constraint": null, "activity_regularizer": null, "activation": "softmax", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_3", "trainable": true, "kernel_regularizer": null}}], "name": "sequential_1"}, "keras_version": "2.4.0", "batch_input_shape": null, "is_graph_network": true, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 5}, "max_ndim": null, "min_ndim": 3}}, "name": "sequential_1", "backend": "tensorflow", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 5]}}
▌


kernel
bias
	keras_api
regularization_losses
	variables
trainable_variables
}__call__
*~&call_and_return_all_conditional_losses"╕	
_tf_keras_layerЮ	{"config": {"dtype": "float32", "use_bias": true, "strides": {"class_name": "__tuple__", "items": [1]}, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "activity_regularizer": null, "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "groups": 1, "filters": 32, "bias_regularizer": null, "activation": "relu", "name": "conv1d_2", "kernel_constraint": null, "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 5]}, "name": "conv1d_2", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 5}, "max_ndim": null, "min_ndim": 3}}, "class_name": "Conv1D", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 5]}, "trainable": true}
ч	

kernel
bias
	keras_api
regularization_losses
	variables
trainable_variables
__call__
+А&call_and_return_all_conditional_losses"┴
_tf_keras_layerз{"config": {"dtype": "float32", "use_bias": true, "strides": {"class_name": "__tuple__", "items": [1]}, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "activity_regularizer": null, "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "groups": 1, "filters": 32, "bias_regularizer": null, "activation": "relu", "name": "conv1d_3", "kernel_constraint": null, "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": null, "name": "conv1d_3", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 32}, "max_ndim": null, "min_ndim": 3}}, "class_name": "Conv1D", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 32]}, "trainable": true}
ч
	keras_api
regularization_losses
	variables
trainable_variables
Б__call__
+В&call_and_return_all_conditional_losses"╓
_tf_keras_layer╝{"must_restore_from_config": false, "batch_input_shape": null, "name": "dropout_1", "config": {"trainable": true, "seed": null, "rate": 0.5, "noise_shape": null, "name": "dropout_1", "dtype": "float32"}, "class_name": "Dropout", "expects_training_arg": true, "dtype": "float32", "stateful": false, "trainable": true}
√
	keras_api
regularization_losses
 	variables
!trainable_variables
Г__call__
+Д&call_and_return_all_conditional_losses"ъ
_tf_keras_layer╨{"config": {"trainable": true, "strides": {"class_name": "__tuple__", "items": [2]}, "data_format": "channels_last", "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "dtype": "float32", "name": "max_pooling1d_1"}, "must_restore_from_config": false, "batch_input_shape": null, "name": "max_pooling1d_1", "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "axes": {}, "max_ndim": null, "min_ndim": null}}, "class_name": "MaxPooling1D", "expects_training_arg": false, "dtype": "float32", "stateful": false, "trainable": true}
ш
"	keras_api
#regularization_losses
$	variables
%trainable_variables
Е__call__
+Ж&call_and_return_all_conditional_losses"╫
_tf_keras_layer╜{"config": {"name": "flatten_1", "trainable": true, "data_format": "channels_last", "dtype": "float32"}, "must_restore_from_config": false, "batch_input_shape": null, "name": "flatten_1", "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {}, "max_ndim": null, "min_ndim": 1}}, "class_name": "Flatten", "expects_training_arg": false, "dtype": "float32", "stateful": false, "trainable": true}
ї

&kernel
'bias
(	keras_api
)regularization_losses
*	variables
+trainable_variables
З__call__
+И&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"config": {"dtype": "float32", "use_bias": true, "units": 100, "kernel_constraint": null, "activity_regularizer": null, "activation": "relu", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_2", "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": null, "name": "dense_2", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 192}, "max_ndim": null, "min_ndim": 2}}, "class_name": "Dense", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 192]}, "trainable": true}
Ў

,kernel
-bias
.	keras_api
/regularization_losses
0	variables
1trainable_variables
Й__call__
+К&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"config": {"dtype": "float32", "use_bias": true, "units": 2, "kernel_constraint": null, "activity_regularizer": null, "activation": "softmax", "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "name": "dense_3", "kernel_regularizer": null}, "must_restore_from_config": false, "batch_input_shape": null, "name": "dense_3", "stateful": false, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "axes": {"-1": 100}, "max_ndim": null, "min_ndim": 2}}, "class_name": "Dense", "expects_training_arg": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "trainable": true}
у
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy"
	optimizer
╩

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
Лserving_default"
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
%:# 2conv1d_2/kernel
: 2conv1d_2/bias
н
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
%:#  2conv1d_3/kernel
: 2conv1d_3/bias
п
Anon_trainable_variables
trainable_variables
Blayer_metrics
regularization_losses
	variables

Clayers
Dlayer_regularization_losses
Emetrics
'А"call_and_return_conditional_losses
__call__
+А&call_and_return_all_conditional_losses"
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
░
Fnon_trainable_variables
trainable_variables
Glayer_metrics
regularization_losses
	variables

Hlayers
Ilayer_regularization_losses
Jmetrics
'В"call_and_return_conditional_losses
Б__call__
+В&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Knon_trainable_variables
!trainable_variables
Llayer_metrics
regularization_losses
 	variables

Mlayers
Nlayer_regularization_losses
Ometrics
'Д"call_and_return_conditional_losses
Г__call__
+Д&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Pnon_trainable_variables
%trainable_variables
Qlayer_metrics
#regularization_losses
$	variables

Rlayers
Slayer_regularization_losses
Tmetrics
'Ж"call_and_return_conditional_losses
Е__call__
+Ж&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
!:	└d2dense_2/kernel
:d2dense_2/bias
░
Unon_trainable_variables
+trainable_variables
Vlayer_metrics
)regularization_losses
*	variables

Wlayers
Xlayer_regularization_losses
Ymetrics
'И"call_and_return_conditional_losses
З__call__
+И&call_and_return_all_conditional_losses"
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
 :d2dense_3/kernel
:2dense_3/bias
░
Znon_trainable_variables
1trainable_variables
[layer_metrics
/regularization_losses
0	variables

\layers
]layer_regularization_losses
^metrics
'К"call_and_return_conditional_losses
Й__call__
+К&call_and_return_all_conditional_losses"
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
╗
	atotal
	bcount
c	keras_api
d	variables"Д
_tf_keras_metricj{"class_name": "Mean", "config": {"name": "loss", "dtype": "float32"}, "dtype": "float32", "name": "loss"}
 
	etotal
	fcount
g
_fn_kwargs
h	keras_api
i	variables"╕
_tf_keras_metricЭ{"class_name": "MeanMetricWrapper", "config": {"fn": "categorical_accuracy", "name": "accuracy", "dtype": "float32"}, "dtype": "float32", "name": "accuracy"}
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
*:( 2Adam/conv1d_2/kernel/m
 : 2Adam/conv1d_2/bias/m
*:(  2Adam/conv1d_3/kernel/m
 : 2Adam/conv1d_3/bias/m
&:$	└d2Adam/dense_2/kernel/m
:d2Adam/dense_2/bias/m
%:#d2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
*:( 2Adam/conv1d_2/kernel/v
 : 2Adam/conv1d_2/bias/v
*:(  2Adam/conv1d_3/kernel/v
 : 2Adam/conv1d_3/bias/v
&:$	└d2Adam/dense_2/kernel/v
:d2Adam/dense_2/bias/v
%:#d2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
щ2ц
 __inference__wrapped_model_93283┴
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *1в.
,К)
conv1d_2_input         
■2√
,__inference_sequential_1_layer_call_fn_93663
,__inference_sequential_1_layer_call_fn_93642
,__inference_sequential_1_layer_call_fn_93542
,__inference_sequential_1_layer_call_fn_93590└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
G__inference_sequential_1_layer_call_and_return_conditional_losses_93466
G__inference_sequential_1_layer_call_and_return_conditional_losses_93768
G__inference_sequential_1_layer_call_and_return_conditional_losses_93719
G__inference_sequential_1_layer_call_and_return_conditional_losses_93493└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_conv1d_2_layer_call_fn_93793в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv1d_2_layer_call_and_return_conditional_losses_93784в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_conv1d_3_layer_call_fn_93802в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv1d_3_layer_call_and_return_conditional_losses_93818в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Р2Н
)__inference_dropout_1_layer_call_fn_93828
)__inference_dropout_1_layer_call_fn_93823┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├
D__inference_dropout_1_layer_call_and_return_conditional_losses_93845
D__inference_dropout_1_layer_call_and_return_conditional_losses_93840┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
К2З
/__inference_max_pooling1d_1_layer_call_fn_93298╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
е2в
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_93295╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
╙2╨
)__inference_flatten_1_layer_call_fn_93850в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_flatten_1_layer_call_and_return_conditional_losses_93856в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_dense_2_layer_call_fn_93865в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_2_layer_call_and_return_conditional_losses_93876в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_dense_3_layer_call_fn_93896в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_3_layer_call_and_return_conditional_losses_93887в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
9B7
#__inference_signature_wrapper_93621conv1d_2_inputЮ
 __inference__wrapped_model_93283z&',-;в8
1в.
,К)
conv1d_2_input         
к "1к.
,
dense_3!К
dense_3         л
C__inference_conv1d_2_layer_call_and_return_conditional_losses_93784d3в0
)в&
$К!
inputs         
к ")в&
К
0          
Ъ Г
(__inference_conv1d_2_layer_call_fn_93793W3в0
)в&
$К!
inputs         
к "К          л
C__inference_conv1d_3_layer_call_and_return_conditional_losses_93818d3в0
)в&
$К!
inputs          
к ")в&
К
0          
Ъ Г
(__inference_conv1d_3_layer_call_fn_93802W3в0
)в&
$К!
inputs          
к "К          г
B__inference_dense_2_layer_call_and_return_conditional_losses_93876]&'0в-
&в#
!К
inputs         └
к "%в"
К
0         d
Ъ {
'__inference_dense_2_layer_call_fn_93865P&'0в-
&в#
!К
inputs         └
к "К         dв
B__inference_dense_3_layer_call_and_return_conditional_losses_93887\,-/в,
%в"
 К
inputs         d
к "%в"
К
0         
Ъ z
'__inference_dense_3_layer_call_fn_93896O,-/в,
%в"
 К
inputs         d
к "К         м
D__inference_dropout_1_layer_call_and_return_conditional_losses_93840d7в4
-в*
$К!
inputs          
p
к ")в&
К
0          
Ъ м
D__inference_dropout_1_layer_call_and_return_conditional_losses_93845d7в4
-в*
$К!
inputs          
p 
к ")в&
К
0          
Ъ Д
)__inference_dropout_1_layer_call_fn_93823W7в4
-в*
$К!
inputs          
p
к "К          Д
)__inference_dropout_1_layer_call_fn_93828W7в4
-в*
$К!
inputs          
p 
к "К          е
D__inference_flatten_1_layer_call_and_return_conditional_losses_93856]3в0
)в&
$К!
inputs          
к "&в#
К
0         └
Ъ }
)__inference_flatten_1_layer_call_fn_93850P3в0
)в&
$К!
inputs          
к "К         └╙
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_93295ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ к
/__inference_max_pooling1d_1_layer_call_fn_93298wEвB
;в8
6К3
inputs'                           
к ".К+'                           ┴
G__inference_sequential_1_layer_call_and_return_conditional_losses_93466v&',-Cв@
9в6
,К)
conv1d_2_input         
p

 
к "%в"
К
0         
Ъ ┴
G__inference_sequential_1_layer_call_and_return_conditional_losses_93493v&',-Cв@
9в6
,К)
conv1d_2_input         
p 

 
к "%в"
К
0         
Ъ ╣
G__inference_sequential_1_layer_call_and_return_conditional_losses_93719n&',-;в8
1в.
$К!
inputs         
p

 
к "%в"
К
0         
Ъ ╣
G__inference_sequential_1_layer_call_and_return_conditional_losses_93768n&',-;в8
1в.
$К!
inputs         
p 

 
к "%в"
К
0         
Ъ Щ
,__inference_sequential_1_layer_call_fn_93542i&',-Cв@
9в6
,К)
conv1d_2_input         
p

 
к "К         Щ
,__inference_sequential_1_layer_call_fn_93590i&',-Cв@
9в6
,К)
conv1d_2_input         
p 

 
к "К         С
,__inference_sequential_1_layer_call_fn_93642a&',-;в8
1в.
$К!
inputs         
p

 
к "К         С
,__inference_sequential_1_layer_call_fn_93663a&',-;в8
1в.
$К!
inputs         
p 

 
к "К         ┤
#__inference_signature_wrapper_93621М&',-MвJ
в 
Cк@
>
conv1d_2_input,К)
conv1d_2_input         "1к.
,
dense_3!К
dense_3         