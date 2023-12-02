object LearningScala1 {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(133); 
  // VALUES are immutable constants. You can't change them once defined.
  val hello: String = "Hi hello OK!";System.out.println("""hello  : String = """ + $show(hello ));$skip(51); 
  //hello = hello + "should work"
  println(hello);$skip(178); 
  
  // Notice how Scala defines things backwards from other languages - you declare the
  // name, then the type.
  
  // VARIABLES are mutable
  var helloThere: String = hello;System.out.println("""helloThere  : String = """ + $show(helloThere ));$skip(33); 
  helloThere = hello + " There!";$skip(22); 
  println(helloThere);$skip(302); 
  
  
  // One key objective of functional programming is to use immutable objects as often as possible.
  // Try to use operations that transform immutable objects into a new immutable object.
  // For example, we could have done the same thing like this:
  val immutableHelloThere = hello + "There!";System.out.println("""immutableHelloThere  : String = """ + $show(immutableHelloThere ));$skip(31); 
  println(immutableHelloThere);$skip(51); 
  
  // Some other types
  val numberOne : Int = 1;System.out.println("""numberOne  : Int = """ + $show(numberOne ));$skip(29); 
  val truth : Boolean = true;System.out.println("""truth  : Boolean = """ + $show(truth ));$skip(27); 
  val letterA : Char = 'a';System.out.println("""letterA  : Char = """ + $show(letterA ));$skip(31); 
  val pi : Double = 3.14159265;System.out.println("""pi  : Double = """ + $show(pi ));$skip(46); 
  val piSinglePrecision : Float = 3.14159265f;System.out.println("""piSinglePrecision  : Float = """ + $show(piSinglePrecision ));$skip(37); 
  val bigNumber : Long = 1234567890l;System.out.println("""bigNumber  : Long = """ + $show(bigNumber ));$skip(31); 
  val smallNumber : Byte = 127;System.out.println("""smallNumber  : Byte = """ + $show(smallNumber ));$skip(141); 
  
  // String printing tricks
  // Concatenating stuff with +:
  println("Here is a mess: " + numberOne + truth + letterA + pi + bigNumber);$skip(71); 
  
  // printf style:
  println(f"Pi is about $piSinglePrecision%.3f");$skip(55); 
  println(f"Zero padding on the left: $numberOne%05d");$skip(169); 
                                                  
  // Substituting in variables:
  println(s"I can use the s prefix to use variables like $numberOne $truth $letterA");$skip(150); 
  // Substituting expressions (with curly brackets):
  println(s"The s prefix isn't limited to variables; I can include any expression. Like ${1+2}");$skip(163); 
                                                 
  // Using regular expressions:
  val theUltimateAnswer: String = "To life, the universe, and everything is 42.";System.out.println("""theUltimateAnswer  : String = """ + $show(theUltimateAnswer ));$skip(37); 
  val pattern = """.* ([\d]+).*""".r;System.out.println("""pattern  : scala.util.matching.Regex = """ + $show(pattern ));$skip(48); 
  val pattern(answerString) = theUltimateAnswer;System.out.println("""answerString  : String = """ + $show(answerString ));$skip(34); 
  val answer = answerString.toInt;System.out.println("""answer  : Int = """ + $show(answer ));$skip(18); 
  println(answer);$skip(54); 
  
  // Dealing with booleans
  val isGreater = 1 > 2;System.out.println("""isGreater  : Boolean = """ + $show(isGreater ));$skip(23); 
  val isLesser = 1 < 2;System.out.println("""isLesser  : Boolean = """ + $show(isLesser ));$skip(40); 
  val impossible = isGreater & isLesser;System.out.println("""impossible  : Boolean = """ + $show(impossible ));$skip(41); 
  val anotherWay = isGreater && isLesser;System.out.println("""anotherWay  : Boolean = """ + $show(anotherWay ));$skip(35); 
  
  val picard: String = "Picard";System.out.println("""picard  : String = """ + $show(picard ));$skip(37); 
  val bestCaptain: String = "Picard";System.out.println("""bestCaptain  : String = """ + $show(bestCaptain ));$skip(46); 
  val isBest: Boolean = picard == bestCaptain;System.out.println("""isBest  : Boolean = """ + $show(isBest ));$skip(34); 
  
  val z = new Array[String](3);System.out.println("""z  : Array[String] = """ + $show(z ));$skip(47); 
  z(0) = "car"; z(1) = "bus"; z(4/2) = "train";$skip(19); 
  println(z(10/5))}
  // EXERCISE
  // Write some code that takes the value of pi, doubles it, and then prints it within a string with
  // three decimal places of precision to the right.
}
