package Training.Kafka.Core;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;

import org.apache.kafka.clients.producer.Callback;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.RecordMetadata;
import org.apache.kafka.common.serialization.StringSerializer;
import org.apache.log4j.PropertyConfigurator;

import Training.Kafka.*;

public class Producer00 
{
	public static void main(String[] args) throws InterruptedException, ExecutionException, IOException 
	{
		PropertyConfigurator.configure("/home/prathamos/Downloads/Kafka/resources/log4j.properties");
		new Producer00().run();
	}
	
    public void run() throws InterruptedException, ExecutionException
    {
    	Properties Config = new Properties();
        
        Config.setProperty(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG,"91.203.135.66:3400,91.203.134.231:3400,91.203.134.181:3400,91.203.133.221:3400");
        Config.setProperty(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,StringSerializer.class.getName());
        Config.setProperty(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,StringSerializer.class.getName());
 
        KafkaProducer<String, String> KP = new KafkaProducer<String, String>(Config);
        
        for(int Counter = 1;Counter<1500;Counter++)
        {
	        ProducerRecord<String, String> PR = 
	        		new ProducerRecord<String, String>("dbs1",
	        				"Message No "+ Integer.toString(Counter) +" From Code");//{"9810","hi"}
	        //KP.send(PR).get();
	        //new MetricsProducerReporter(KP).run();
	        KP.send(PR, new Callback() {
				public void onCompletion(RecordMetadata RMD, Exception Ex) {
					if(Ex == null)
					{
						System.out.println("Info Recieved: \n"+
								"Partition:"+RMD.partition()+ "\n"+
								"Offset:"+RMD.offset());
					}
					else {System.out.println("Error : "+Ex.toString());}
				}
	        });
        }
        KP.flush();
        KP.close();
    }
}
