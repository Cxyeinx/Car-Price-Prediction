	?G??|6@?G??|6@!?G??|6@	????e@????e@!????e@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?G??|6@8I?Ǵ6??A/i??Qu@Y?Bus????rEagerKernelExecute 0*	cX9?@l@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!?V??~O@)mo?$???1%?=??>M@:Preprocessing2U
Iterator::Model::ParallelMapV2??&????!??5??f+@)??&????1??5??f+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat^-wf???!5????B.@)?vR~R??1S?2??V)@:Preprocessing2F
Iterator::Model\??J?H??!?Cxd2@)?d???1???/??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??S ?g??!?ݶ?Y@)??S ?g??1?ݶ?Y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?t["???!/z??fT@)[?kBZc??1?P?R@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorq:?v?!?kם??@)q:?v?1?kם??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ai?G5??!?????wO@)$
-???`?1??L????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????e@I? ?tX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	8I?Ǵ6??8I?Ǵ6??!8I?Ǵ6??      ??!       "      ??!       *      ??!       2	/i??Qu@/i??Qu@!/i??Qu@:      ??!       B      ??!       J	?Bus?????Bus????!?Bus????R      ??!       Z	?Bus?????Bus????!?Bus????b      ??!       JCPU_ONLYY????e@b q? ?tX@