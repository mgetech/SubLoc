import numpy as np
import codecs
from feedback import pre
def my_gen(sess, trainable_model, batch_size, generated_num, output_file):
    print('!!!!!!!!!!!!!!!!!!!!making my datas!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    
#    print(f"genearated sample: {generated_samples}")

    generated_samples_,negative = pre(generated_samples,0.65)
    print(f"generated_samples_ len: {len(generated_samples_)}")
    print(f"negative len: {len(negative)}")
    
    with open("finaly_0.65.txt", "a") as f:
        f.write(str(len(generated_samples_))+' '+str(len(negative)))
        f.write("\n")
        
    print(f"genearated sample_: {generated_samples_}")
    with codecs.open(output_file, 'a+', 'utf-8') as fout:
        for poem in generated_samples_:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)
