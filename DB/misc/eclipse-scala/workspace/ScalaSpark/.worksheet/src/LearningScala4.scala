object LearningScala4 {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(304); 
  // Data structures
  
  // Tuples (Also really common with Spark!!)
  // Immutable lists
  // Often thought of as database fields, or columns.
  // Useful for passing around entire rows of data.
  
  val captainStuff = ("Picard", "Enterprise-D", "NCC-1701-D", ("crew1","crew2"));System.out.println("""captainStuff  : (String, String, String, (String, String)) = """ + $show(captainStuff ));$skip(24); 
  println(captainStuff);$skip(94); 
  
  // You refer to individual fields with their ONE-BASED index:
  println(captainStuff._1);$skip(27); 
  println(captainStuff._2);$skip(27); 
  println(captainStuff._3);$skip(30); 
  println(captainStuff._4._2);$skip(90); 
 // You can create a key/value pair with ->
 val picardsShip = "Picard" -> "Enterprise-D";System.out.println("""picardsShip  : (String, String) = """ + $show(picardsShip ));$skip(25); 
 println(picardsShip._2);$skip(87); 
 
 // You can mix different types in a tuple
 val aBunchOfStuff = ("Kirk", 1964, true);System.out.println("""aBunchOfStuff  : (String, Int, Boolean) = """ + $show(aBunchOfStuff ));$skip(275); 
 
 
 
 // Lists
 // Like a tuple, but it's an actual Collection object that has more functionality.
 // Also, it cannot hold items of different types.
 // It's a singly-linked list under the hood.
 
 val shipList = List("Enterprise", "Defiant", "Voyager", "Deep Space Nine");System.out.println("""shipList  : List[String] = """ + $show(shipList ));$skip(101); 
 
 // Access individual members using () with ZERO-BASED index (confused yet?)
 println(shipList(1));$skip(93); 
 
 // head and tail give you the first item, and the remaining ones.
 println(shipList.head);$skip(24); 
 println(shipList.tail);$skip(72); 
 
 
 // Iterating though a list
 for (ship <- shipList) {println(ship)};$skip(187); 
 
 // Let's apply a function literal to a list! map() can be used to apply any function to every item in a collection.
val backwardShips = shipList.map( (ship: String) => {ship.reverse});System.out.println("""backwardShips  : List[String] = """ + $show(backwardShips ));$skip(45); 
 for (ship <- backwardShips) {println(ship)};$skip(183); 
                                                  
// reduce() can be used to combine together all the items in a collection using some function.
val numberList = List(1, 2, 3, 4, 5);System.out.println("""numberList  : List[Int] = """ + $show(numberList ));$skip(56); 
val sum = numberList.reduce( (x: Int, y: Int) => x + y);System.out.println("""sum  : Int = """ + $show(sum ));$skip(13); 
println(sum);$skip(158); 

// filter() can remove stuff you don't want. Here we'll introduce wildcard syntax while we're at it.
val iHateFives = numberList.filter( (x: Int) => x != 5);System.out.println("""iHateFives  : List[Int] = """ + $show(iHateFives ));$skip(44); 
val iHateThrees = numberList.filter(_ != 3);System.out.println("""iHateThrees  : List[Int] = """ + $show(iHateThrees ));$skip(229); 

// Note that Spark has its own map, reduce, and filter functions that can distribute these operations. But they work the same way!
// Also, you understand MapReduce now :)

// Concatenating lists
val moreNumbers = List(6, 7, 8);System.out.println("""moreNumbers  : List[Int] = """ + $show(moreNumbers ));$skip(46); 
val lotsOfNumbers = numberList ++ moreNumbers;System.out.println("""lotsOfNumbers  : List[Int] = """ + $show(lotsOfNumbers ));$skip(52); 

// More list fun
val reversed = numberList.reverse;System.out.println("""reversed  : List[Int] = """ + $show(reversed ));$skip(29); 
val sorted = reversed.sorted;System.out.println("""sorted  : List[Int] = """ + $show(sorted ));$skip(48); 
val lotsOfDuplicates = numberList ++ numberList;System.out.println("""lotsOfDuplicates  : List[Int] = """ + $show(lotsOfDuplicates ));$skip(47); 
val distinctValues = lotsOfDuplicates.distinct;System.out.println("""distinctValues  : List[Int] = """ + $show(distinctValues ));$skip(30); 
val maxValue = numberList.max;System.out.println("""maxValue  : Int = """ + $show(maxValue ));$skip(27); 
val total = numberList.sum;System.out.println("""total  : Int = """ + $show(total ));$skip(39); 
val hasThree = iHateThrees.contains(3);System.out.println("""hasThree  : Boolean = """ + $show(hasThree ));$skip(223); 

// Maps
// Useful for key/value lookups on distinct keys
// Like dictionaries in other languages

val shipMap = Map("Kirk" -> "Enterprise", "Picard" -> "Enterprise-D", "Sisko" -> "Deep Space Nine", "Janeway" -> "Voyager");System.out.println("""shipMap  : scala.collection.immutable.Map[String,String] = """ + $show(shipMap ));$skip(28); 
println(shipMap("Janeway"));$skip(66); 

// Dealing with missing keys
println(shipMap.contains("Archer"));$skip(67); 

val archersShip = util.Try(shipMap("Archer")) getOrElse "Unknown";System.out.println("""archersShip  : String = """ + $show(archersShip ));$skip(21); 
println(archersShip)}

// EXERCISE
// Create a list of the numbers 1-20; your job is to print out numbers that are evenly divisible by three. (Scala's
// modula operator, like other languages, is %, which gives you the remainder after division. For example, 9 % 3 = 0
// because 9 is evenly divisible by 3.) Do this first by iterating through all the items in the list and testing each
// one as you go. Then, do it again by using a filter function on the list instead.

}
