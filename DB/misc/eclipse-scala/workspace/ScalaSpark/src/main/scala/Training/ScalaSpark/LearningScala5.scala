import java.io._
import scala.collection.mutable.ArrayBuffer

class Student(val Id: Int, val FirstName: String) {
  var x: Int = Id
  var y: String = FirstName
  private var _gmatscore = 70
  //getter setter
  def getgmatscore = _gmatscore
  def setgmatscore_=(value: Int): Unit = _gmatscore = value

  def PrintInfo(NewId: Int, LastName: String) {
    x = x + NewId
    y = y + " " + LastName
    println("NewId : " + x);
    println("FullName : " + y);
  }
}
//constructor and aux
class Employee {
  private var name = ""
  private var project = ""

  def this(name: String) {
    this()
    this.name = name
  }

  def this(name: String, age: Int) {
    this(name)
    this.project = project
  }
}
//classes cannot have static variables or methods. Instead a singleton object / companion object.
object Student {
def getStrategy(enoughEnergy: Boolean) = {
 if (enoughEnergy)
 (energy: Double) => println("We are going to attack with damage "+energy)
 else
 (energy: Double) => println("We are going to reflect damage "+energy/2)
}
  
  def main(args: Array[String]) {
    var st = new Student(10, "Rakesh");

    st.PrintInfo(10, "Khandelwal");
    println("original gmatscore : " + st.getgmatscore);
    st.setgmatscore_=(95);
    println("new gmatscore : " + st.getgmatscore);

    CompanionExample.sayHello();

    var _human = new human()
    _human.whoami()
    _human.hi()
    _human.hi("rakesh")

    var h = new qwer(10)
    h.method1()
    h.method2()

    val e = "45"
    val f = e.asInstanceOf[String].toInt
    println(f * 10)

    /*array buffer start*/
    val arrayBuffer1: ArrayBuffer[String] = ArrayBuffer("cat", "dog", "mouse")
    //> arrayBuffer1  : scala.collection.mutable.ArrayBuffer[String] = ArrayBuffer(c
    //| at, dog, mouse)
    println(s"Elements of arrayBuffer1 = $arrayBuffer1")
    //> Elements of arrayBuffer1 = ArrayBuffer(cat, dog, mouse)

    println(s"Element at index 0 = ${arrayBuffer1(0)}")
    //> Element at index 0 = cat

    arrayBuffer1 += "horse"
    /*array buffer end*/

    /*case class start*/
    val alice = new Person("Alice", 25)
    val bob = new Person("Bob", 32)
    val charlie = new Person("Charlie", 32)

    for (person <- List(alice, bob, charlie)) {
      person match {
        case Person("Alice", 25) => println("Hi Alice!")
        case Person("Bob", 32)   => println("Hi Bob!")
        case Person(name, age) =>
          println("Age: " + age + " year, name: " + name + "?")
      }
    }
    /*case class end*/

    /*nested class start*/
    val a1 = new Calculator
    val a2 = new Calculator
    val b1 = new a1.Add2Num
    val b2 = new a2.Add2Num

    b1.a = 30;
    b1.b = 45;
    b2.a = 55;
    b2.b = 24;    
    println(s"b1.a = ${b1.a}")
    println(s"b1.b = ${b1.b}")
    println(s"b2.a = ${b2.a}")
    println(s"b2.b = ${b2.b}")
    println(s"Result = ${b2.c}")
    b2.Add();
    println(s"Result = ${b2.c}")
    /*nested class end*/

    /* higher order function return function*/
    val returnedFunction = getStrategy(true)
    returnedFunction(15.0) 
    
    //A closure is a function, whose return value depends on the value of one or more variables declared outside this function.
    var factor = 3
    val multiplier = (i:Int) => i * factor
    val multiplier2 = (i:Int) => i * 10
    println( "multiplier(1) value = " +  multiplier(5) )
    println( "multiplier(2) value = " +  multiplier2(2) )
    
    var obj = new qwer(545)
    obj.method1()
    obj.method2()
    obj.somemethod("A")
    obj.display("B")
    obj.show("C")
  }
}
//case classes
case class Person(name: String, age: Int)

//abstract class
abstract class asdf(a: Int) { // Creating constructor
  var b: Int = 20 // Creating variables
  var c: Int = 25
  def method1() // Abstract method
  def method2() { // Non-abstract method
    println("method2")
  }
}

trait trait0{
    def a = "from trait0"
}

trait trait1 extends trait0{
    def somemethod(x: String): String
    
    def display(mesage: String) {
      println("trait1 message " + mesage + " "+ super.a)
    }    
}  


trait trait2 extends trait0{
    def show(mesage: String) {
      println("trait2 message " + mesage + " "+ super.a)
    }
}


class qwer(a: Int) extends asdf(a) with trait1  with trait2 {
  c = 30
  def method1() {
    println("method1...")
    println("a = " + a)
    println("b = " + b)
    println("c = " + c)
  }
def somemethod(x: String): String = {
  	"SOME METHOD " + x
  }
}



class Calculator {
  class Add2Num {
    var a = 12;
    var b = 31;
    var c = a + b;
    
    def Add() {
        c = a + b;
      }    
  }
}

object CompanionExample {
  def sayHello() {
    println("hello universe");
  }
}

//method overriding & overloading
class mammal {
  def whoami() {
    println("i am a mammal")
  }
}
class human extends mammal {
  override def whoami() {
    println("i am a human")
  }
  def hi() {
    println("hi there")
  }
  def hi(name: String) {
    println("hi there " + name)
  }
}
//https://stackoverflow.com/questions/7484928/what-does-a-lazy-val-do
//https://medium.com/@kadirmalak/currying-in-scala-a-useful-example-bd0e3a44195
//https://www.baeldung.com/scala/foldleft-vs-reduceleft
//https://www.baeldung.com/scala/sorting