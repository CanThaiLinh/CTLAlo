//
//  ViewController.swift
//  StockPrices
//
//  Created by thailinh on 5/13/20.
//  Copyright © 2020 thailinh. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    //    var stock_price = [10,7,100,5,8,11,9];
    var stock_price = [10,7,5];
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        //        cal()
        
        //        let trust = [[1,3],[1,4],[2,3],[2,4],[4,3]]
        //        let n = 4
        //        let value = findJudge(n, trust)
        //        print(value)
        
        //        let image = [[0,0,0],[0,1,0]]
        //        let sr = 1, sc = 1, newColor = 2
        //        let newImage = floodFill(image, sr, sc, newColor)
        //        print(newImage)
        
        //        let num = "1230", k = 3
        //        let result  = removeKdigits(num, k)
        //        print(result)
        //        totalNQueens(8)
//        testTrie()
        print("test")

//        let ans = findAnagrams("cbaebabacd", "abc")
//        for item in ans{
//            print(item)
//        }
//        let num1 = [0,2,5,7,8,12222]
//        let num2 = [0,2,3,4,5,6]
//        let ans = findMedianSortedArrays(num1, num2)
//        print("ans findMedianSortedArrays \(ans)")
        let st = StockSpanner()
        st.next(100)
        st.next(80)
        st.next(60)
        st.next(70)
        st.next(60)
        st.next(75)
        st.next(85)
        print("test end")
    }
    func quickSort(list : inout [Int], left : Int, right : Int){
        let index = left  + (right - left + 1) / 2
        let pivot = list[ index]
        var newLeft = left, newRight = right
        while newLeft < newRight {
            while  list[newLeft] < pivot {
                newLeft += 1
            }
            while list[newRight] > pivot {
                newRight -= 1
            }
            if  newLeft <= newRight{
                list.swapAt(newLeft, newRight)
                newRight -= 1
                newLeft += 1
            }
        }
        if left < newRight{
            quickSort(list: &list, left: left, right: newRight)
        }
        if newLeft < right{
            quickSort(list: &list, left: newLeft, right: right)
        }
        
    }
    func cal(){
        guard stock_price.count > 2 else{
            print("deo duoc")
            return;
        }
        var minPrice = stock_price[0]
        var profit = stock_price[1] - stock_price[0]
        for index in 1..<stock_price.count{
            let value = stock_price[index]
            profit = max(profit, value - minPrice)
            minPrice = min(minPrice, value)
        }
        print("minprice = \(minPrice) and profit = ", profit)
        var num = 100
        for index in 2...num/2{
            
        }
    }
    func findJudge(_ N: Int, _ trust: [[Int]]) -> Int {
        if N == 1{
            return 1
        }
        var dicTrust = [Int : [Int]]()
        var dicToBeTrust =  [Int : [Int]]()
        var listJudges = [Int]()
        
        
        for list in trust{
            if dicTrust[list[0]] == nil{
                dicTrust[list[0]] = [Int]()
            }
            dicTrust[list[0]]!.append(list[1])
            
            if dicToBeTrust[list[1]] == nil{
                dicToBeTrust[list[1]] = [Int]()
            }
            dicToBeTrust[list[1]]!.append(list[0])
        }
        print("list trust = ",dicTrust)
        print("list tobe trusted = ", dicToBeTrust)
        for index in 1...N{
            if dicTrust[index] == nil || dicTrust[index]!.count == 0{
                listJudges.append(index)
            }
            
        }
        print("listjuged= ", listJudges)
        var result = -1
        for jugdge in listJudges{
            if dicToBeTrust[jugdge]?.count == N - 1{
                if result == -1 {
                    result = jugdge
                }else{
                    return -1
                }
            }
        }
        
        return result
    }
    
    func floodFill(_ image: [[Int]], _ sr: Int, _ sc: Int, _ newColor: Int) -> [[Int]] {
        let currentColor = image[sr][sc]
        if currentColor == newColor{
            return image
        }
        var newImage = image
        newImage[sr][sc] = newColor
        return fillColor(newImage, sr, sc, newColor, currentColor)
    }
    
    func fillColor(_ image: [[Int]], _ sr: Int, _ sc: Int, _ newColor: Int,_ oldColor : Int) -> [[Int]] {
        var newImage = image
        //        print("new image = ",newImage)
        let dx = [-1,1,0,0]
        let dy = [0,0,1,-1]
        for index in 0...3{
            let newX = sr + dx[index]
            let newY = sc + dy[index]
            if newX >= 0 && newX < image.count && newY >= 0 && newY < image[newX].count
                && image[newX][newY] == oldColor
            {
                newImage[newX][newY] = newColor
                newImage = fillColor(newImage, newX, newY, newColor, oldColor)
            }
        }
        return newImage
    }
    
    func singleNonDuplicate(_ nums: [Int]) -> Int {
        var left = 0
        var right = nums.count - 1
        var mid = 0
        while left < right {
            mid = left + (right - left) / 2
            if nums[mid] == nums[mid ^ 1]{
                left = mid + 1
            }else{
                right = mid
            }
        }
        return nums[left]
    }
    
    func removeAllFirstZero( num : inout String){
        while (num.count > 0 && num.first! == "0")  {
            num.removeFirst()
        }
    }
    
    func getCharacterAtIndex(index : Int, string : String) -> Character{
        let indexStr = string.index(string.startIndex, offsetBy: index)
        return string[indexStr]
    }
    func removeKdigits(_ num: String, _ k: Int) -> String {
        var index = 0
        var newNum = num
        var newK = k
        while ( index < newNum.count - 1 && newK > 0) {
            if (getCharacterAtIndex(index: index, string: newNum) <= getCharacterAtIndex(index: index + 1, string: newNum)){
                index += 1
            }else{
                newNum.remove(at: newNum.index(newNum.startIndex, offsetBy: index))
                newK -= 1
                if index > 0{
                    index -= 1
                }
            }
        }
        print(newNum)
        if newK != 0{
            newNum.removeLast(newK)
        }
        if (newNum.count > 0 && newNum.first! == "0"){
            removeAllFirstZero(num: &newNum)
        }
        if newNum.count == 0{
            return "0"
        }
        return newNum
    }
    public class TreeNode {
        public var val: Int
        public var left: TreeNode?
        public var right: TreeNode?
        public init() { self.val = 0; self.left = nil; self.right = nil; }
        public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
        public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
            self.val = val
            self.left = left
            self.right = right
            
        }
    }
    func dfsSearchNode(root : inout TreeNode?, x : Int, y : Int, depth : Int, parentNode : inout TreeNode?, listInfoDepths : inout [Int], listInfoParents : inout [TreeNode]){
        guard var rootNode = root  else {
            return
        }
        if rootNode.val == x || rootNode.val == y{
            listInfoDepths.append(depth)
            if parentNode == nil{
                listInfoParents.append(TreeNode())
            }
            
        }
        dfsSearchNode(root: &rootNode.left, x: x, y: y, depth: depth + 1, parentNode: &root, listInfoDepths: &listInfoDepths, listInfoParents: &listInfoParents)
        dfsSearchNode(root: &rootNode.right, x: x, y: y, depth: depth + 1, parentNode: &root, listInfoDepths: &listInfoDepths, listInfoParents: &listInfoParents)
    }
    func isCousins(_ root: TreeNode?, _ x: Int, _ y: Int) -> Bool {
        var nR = root
        guard var rootNode = root  else {
            return false
        }
        var listInfoDepths = [Int]()
        var listInfoParents = [TreeNode]()
        var treeNil : TreeNode? = nil
        dfsSearchNode(root: &nR, x: x, y: y, depth: 0, parentNode: &treeNil, listInfoDepths: &listInfoDepths, listInfoParents: &listInfoParents)
        if listInfoParents[0].val != 0 && listInfoParents[1].val != 0 {
            return (listInfoDepths[0] == listInfoDepths[1]) && (listInfoParents[0].val != listInfoParents[1].val)
        }
        return false
        
    }
    
    func canConstruct(_ ransomNote: String, _ magazine: String) -> Bool {
        if ransomNote.count > magazine.count {
            return false
        }
        if ransomNote.count == 0 && magazine.count == 0 {
            return true
        }
        var mark = [Character : Int]()
        for char in magazine{
            if mark[char] == nil{
                mark[char] = 0
            }
            mark[char]! += 1
        }
        for char in ransomNote{
            if mark[char] == nil || mark[char] == 0{
                return false
            }
            mark[char]! -= 1
        }
        return true
    }
    func firstUniqChar(_ s: String) -> Int {
        var mark = [Character : Int]()
        for char in s{
            if mark[char] == nil{
                mark[char] = 0
            }
            mark[char]! += 1
        }
        for (index, char) in s.enumerated(){
            if mark[char]! == 1{
                return index
            }
        }
        return -1
    }
    func totalNQueens(_ n: Int) -> Int {
        queen2(n)
        return 0
    }
    func queen2(_ n: Int){
        var board = Array(repeating: Array(repeating: 0, count: n), count: n)
        //        print(board)
        
    }
    func testTrie() {
//        var trieObj = Trie()
//        trieObj.insert("apple")
//
//        let isSearch = trieObj.search("app")
//        print("isSearch = \(isSearch ? "true" : "false")")
    }
    
    
    
    // kadane algorithm dynamic program
    func findMaxSumSubArray(list : [Int]) -> Int{
        var maxSumTemp = 0
        var maxSum = Int.min
        for item in list{
            maxSumTemp += item
            if maxSum < maxSumTemp{
                maxSum = maxSumTemp
            }
            maxSumTemp = max(maxSumTemp,0)
        }
        return maxSum
    }
    func maxSubarraySumCircular(_ A: [Int]) -> Int {
        let maxOfInitialList = findMaxSumSubArray(list: A)
        var total = 0
        for item in A{
            total += item
        }
        var listA2 = [Int]()
        listA2.append(contentsOf: A)
        
        for index in 0..<listA2.count{
            listA2[index] = A[index] * -1
        }
        let maxTemp = findMaxSumSubArray(list: listA2)
        let flipSum = total + maxTemp
        if flipSum > maxOfInitialList && flipSum != 0 {
            return flipSum
        }
        return maxOfInitialList
        
    }
    public class ListNode {
         public var val: Int
         public var next: ListNode?
         public init() { self.val = 0; self.next = nil; }
         public init(_ val: Int) { self.val = val; self.next = nil; }
        public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
    }
    
    func oddEvenList(_ head: ListNode?) -> ListNode? {
        guard var oddHead = head else{return nil}
        var even = head?.next
        var evenHead = even
        while ( even != nil && even?.next != nil ){
            oddHead.next = even?.next
            oddHead = oddHead.next!
            even?.next = oddHead.next
            even = even?.next
        }
        oddHead.next = evenHead
        return head
    }
    
    func findAnagrams(_ s: String, _ p: String) -> [Int] {
        var results = [Int]()
        
        let lengthOfP = p.count
        let lengthOfS = s.count
     
        
        if(lengthOfS==0||lengthOfS<lengthOfP){return results}
        
        let newS = s.map { Int($0.asciiValue! - 97) }
        let newP = p.map { Int($0.asciiValue! - 97) }
        
        var markCharactersOfP = Array(repeating: 0, count: 26)
        var markCharactersOfS = Array(repeating: 0, count: 26)
        
        for index in 0..<p.count {
            markCharactersOfP[newP[index]] += 1
            
            markCharactersOfS[newS[index]] += 1
        }
        
        if compareArray(left: markCharactersOfS, right: markCharactersOfP){
            results.append(0)
        }

        for index in (lengthOfP)..<lengthOfS{
            
            markCharactersOfS[newS[index]] += 1
            markCharactersOfS[newS[index - lengthOfP]] -= 1

            if compareArray(left: markCharactersOfS, right: markCharactersOfP){
                results.append(index - lengthOfP + 1)
            }
        }
        
        return results
    }
 
//    func findAnagrams(_ s: String, _ p: String) -> [Int] {
//        func charIndex(_ char : Character) -> Int{
//            return Int(char.asciiValue! - Character("a").asciiValue!)
//        }
//
//
//        var result : [Int] = [Int]()
//        guard s.count > p.count else {
//            return result
//        }
//
//        let s = s.map { charIndex($0) }
//        let p = p.map { charIndex($0) }
//
//        var originalMap = Array(repeating: 0, count: 26)
//        for char in p {
//            originalMap[char] += 1
//        }
//        var map = originalMap.map { _ in 0 }
//        for i in 0..<s.count {
//            map[s[i]] += 1
//            if i - p.count >= 0 {
//                map[s[i-p.count]] -= 1
//            }
//
//            if map == originalMap {
//                result.append(i - p.count + 1)
//            }
//        }
//        return result
//    }
//    func compareDictionary <K, V>(left: [K:V?], right: [K:V?]) -> Bool {
//        guard let left = left as? [K: V], let right = right as? [K: V] else { return false }
//        return NSDictionary(dictionary: left).isEqual(to: right)
//    }
    func compareArray (left : [Int], right :  [Int]) -> Bool {
        // left count == right count
        for index in 0..<left.count{
            if left[index] != right[index]{
                return false
            }
        }
        return true
    }
    
    func findMedianSortedArrays2(_ nums1: [Int], _ nums2: [Int]) -> Double {
        var av : Double = 0.0
        var total = 0
        for item in nums1{
            total += item
        }
        for item in nums2{
            total += item
        }
        return Double(Double(total ) / Double(nums1.count + nums2.count))
    }
    
    func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
        let firstAns = findMedianSortedArrays2(nums1, nums2)
        print("findMedianSortedArrays2  = \(firstAns)")
        let m = nums1.count
        let n = nums2.count
        if m == 0 {
            return Double(nums2[n/2] + nums2[(n-1)/2]) / 2.0
        }
        if n == 0 {
            return Double(nums1[m/2] + nums1[(m-1)/2]) / 2.0
        }
        
        var L = nums1 //larger array
        var S = nums2 //smaller array
        if L.count < S.count {
            L = nums2
            S = nums1
        }
        
        var l = L.count //L size
        var s = S.count //S size
        var left = 0, right = s
        while left <= right {
            var smid = (left + right)/2 //binary search for desired partition in smaller array
            var lmid = (l+s+1) / 2 - smid //balancing partition in larger array

            var sl = smid <= 0 ? Int.min : S[smid-1] //value left of mid in smaller array
            var sr = smid >= s ? Int.max : S[smid] //value right of mid in smaller array

            var ll = lmid <= 0 ? Int.min : L[lmid-1] //value right of mid in larger array
            var lr = lmid >= l ? Int.max : L[lmid] //value right of mid in larger array

            if sl > lr { //we are too right of solution
                right = smid - 1
            } else if ll > sr { //we are too left of solution
                left = smid + 1
            } else { //we found the solution
                if (l+s) % 2 == 1 {
                    return Double(max(sl, ll))
                } else {
                    return Double(min(sr, lr) + max(sl, ll)) / 2.0
                }
            }
        }
        return 0.0 // should not reach for a valid input
    }
    func kthSmallest(_ root: TreeNode?, _ k: Int) -> Int {
        var value : Int? = nil
        var count : Int = 0
        search(root,k, &value, &count)
        return value!
    }
    
    func search(_ root : TreeNode?,_ k : Int,_ value : inout Int?, _ count : inout Int){
        guard let root = root else{
            return
        }
        if let left = root.left{
            search(left,k, &value, &count)
        }
        count += 1
        if count == k{
            value = root.val
            return
        }
        
        if let right = root.right {
            search(right,k, &value, &count)
        }
        
    }
    func countSquares(_ matrix: [[Int]]) -> Int {
        var newMatrix = matrix
        let rowLength = newMatrix.count
        if rowLength == 0 {return 0}
        let colLength = newMatrix[0].count
        var result = 0
        
        for indexRow in 0..<rowLength{
            for indexCol in 0..<colLength{
                // Case : value is zero. Not care
                if newMatrix[indexRow][indexCol] == 0{continue}
                // the value here is one
                // Case : the bolder cell is one. maxmium square = 1. indexRow and indexCol is the bottom right of square to calculate
                if indexRow == 0 || indexCol == 0{
                    result += 1
                    continue
                }
                // Case : inner cell is one.
                // check : left cell, up cell, diagonal. if all is one => add 1 to newMatrix[indexRow][indexCol] and the square is lagger than 1 (value is 2)
                // Do this for all value. we can store the last value and do not calculate duplicate.
                // so we find the minium of left cell, up cell, diagonal cell and add to current cell.
                // indexRow and indexCol is the bottom right of square to calculate
                let upCell = newMatrix[indexRow - 1][indexCol]
                let leftCell = newMatrix[indexRow ][indexCol - 1]
                let diagonalCell = newMatrix[indexRow - 1][indexCol - 1]
                let minValue = min(upCell, min(leftCell , diagonalCell ))
                newMatrix[indexRow][indexCol] += minValue
                result += newMatrix[indexRow ][indexCol]
            }
        }
        return result
    }
    func frequencySort(_ s: String) -> String {
        var result = ""
        var mark = [Character : Int]()
        for item in s {
            if mark[item] == nil{
                mark[item] = 0
            }
            mark[item]! += 1
        }
        var listKeys = Array(mark.keys)
        let sortedCharacters = listKeys.sorted {
            mark[$0] ?? 0 > mark[$1] ?? 0
        }
        
        for item in sortedCharacters{
            let freq = mark[item]!
            for _ in 1...freq{
                result += "\(item)"
            }
        }
        return result
    }
    func bstFromPreorder(_ preorder: [Int]) -> TreeNode? {
        var root : TreeNode?
        for value in preorder{
            bstUtil(root: &root, value: value)
        }
        return root
    }
    
    func bstUtil(root : inout TreeNode?, value : Int)->TreeNode{
        if root == nil{
            root = TreeNode(value)
            return root!
        }
        if root!.val > value{
            root!.left = bstUtil(root: &root!.left, value: value)
        }else{
            root!.right = bstUtil(root: &root!.right, value: value)
        }
        return root!
    }
    func maxUncrossedLines(_ A: [Int], _ B: [Int]) -> Int {
        let aCount = A.count
        let bCount = B.count
        var dp = Array(repeating: Array(repeating: 0, count: bCount + 1), count: aCount + 1)
        for i in 0..<aCount{
            for j in 0..<bCount{
                if (A[i] == B[j]){
                    dp[i+1][j+1] = 1 + dp[i][j]
                }else{
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
                }
            }
        }
        return dp[aCount][bCount]
        
    }
//    func findMaxLength(_ nums: [Int]) -> Int {
//        var maxLength = 0
//
//        for i in 0..<nums.count{
//            var zeroCount = 0, oneCount = 0
//            for j in i..<nums.count{
//                if nums[j] == 0{
//                    zeroCount += 1
//                }else{
//                    oneCount += 1
//                }
//                if zeroCount == oneCount{
//                    maxLength = max(maxLength, j-i + 1)
//                }
//            }
//        }
//        return maxLength
//    }
    func findMaxLength(_ nums: [Int]) -> Int {
        var mark = [Int : Int]()
        mark[0] = -1
        var maxLength = 0, count = 0
        for index in 0..<nums.count{
            count = count + (nums[index] == 1 ? 1 : -1)
            if mark.keys.contains(count){
                maxLength = max(maxLength, index - mark[count]!)
            }else{
                mark[count] = index
            }
        }
        return maxLength
        
    }
//    func possibleBipartition(_ N: Int, _ dislikes: [[Int]]) -> Bool {
//        var markGroupA = [Int : Bool]()
//        var markGroupB = [Int : Bool]()
//        for dislike in dislikes{
//            let firstPerson = dislike[0]
//            let secondPerson = dislike[1]
//            if markGroupA[firstPerson] != nil && markGroupA[firstPerson] == true{
//                if markGroupA[secondPerson] != nil && markGroupA[secondPerson] == true{
//                    return false
//                }else{
//                    markGroupA[firstPerson] = true
//                    markGroupB[secondPerson] = true
//                }
//            }else if markGroupB[firstPerson] != nil && markGroupB[firstPerson] == true{
//                if markGroupB[secondPerson] != nil && markGroupB[secondPerson] == true{
//                    return false
//                }else{
//                    markGroupB[firstPerson] = true
//                    markGroupA[secondPerson] = true
//                }
//            }else{
//                if markGroupA[secondPerson] != nil && markGroupA[secondPerson] == true{
//                    markGroupB[firstPerson] = true
//                    markGroupA[secondPerson] = true
//                }else {
//                    //markGroupB[secondPerson] != nil ||  markGroupB[secondPerson] == true || markGroupB[secondPerson] == false
//                    markGroupA[firstPerson] = true
//                    markGroupB[secondPerson] = true
//                }
//            }
//        }
//        return true
//    }
    func possibleBipartition(_ N: Int, _ dislikes: [[Int]]) -> Bool {
        var graph:[[Int]] = []
        for _ in 0..<N{
            graph.append([])
        }
        for dislike in dislikes{
            let from = dislike[0]-1
            let to = dislike[1]-1
            graph[from].append(to)
            graph[to].append(from)
        }
        var visited = Array(repeating: 0, count: graph.count)
        for i in 0..<N{
            if 0 == visited[i]{
                let res = bfs(i ,graph, &visited)
                if !res{
                    return false
                }
            }
        }
        return true
    }
    
    func bfs(_ node:Int, _ graph:[[Int]], _ visited:inout[Int])->Bool{
        var queue:[Int] = []
        queue.append(node)
        visited[node] = 1

        while !queue.isEmpty{
            let item = queue.removeFirst()
            for child in graph[item]{
                if visited[child] == 0{
                    visited[child] = visited[item] == 1 ? 2 : 1
                    queue.append(child)
                }else{
                    if visited[child] == visited[item] { return false }
                }
            }
        }
        return true
    }
}

/*
struct TrieNode{
    var isword = false
    var next : [TrieNode?] = Array(repeating: nil, count: 26)
}
// TRIE
class Trie {
    private var root : TrieNode!
    func find(s : String) -> TrieNode?{
        var rootTemp = root
        for char in s{
            let index : Int = Int(char.unicodeScalars.first!.value - 97)
            if rootTemp!.next[index] == nil{
                return nil
            }
            
            rootTemp = rootTemp!.next[index]
        }
        return rootTemp
    }
    
    /** Initialize your data structure here. */
    init() {
        root = TrieNode()
    }
    
    /** Inserts a word into the trie. */
    func insert(_ word: String) {
        insertRecursive(node: &root!, index: 0, word: word)
    }
    func insertRecursive(node : inout TrieNode, index : Int, word : String){
        guard index < word.count else{
            node.isword = true
            return
        }
        
        let char = getCharacterAtIndex(index: index, string: word)
        let idx : Int = Int(char.unicodeScalars.first!.value - 97)

        if (node.next[idx] == nil){
            node.next[idx] = TrieNode()
        }
        insertRecursive(node: &node.next[idx]!, index: index + 1, word: word)
        
    }
    func getCharacterAtIndex(index : Int, string : String) -> Character{
        let indexStr = string.index(string.startIndex, offsetBy: index)
        return string[indexStr]
    }
    /** Returns if the word is in the trie. */
    func search(_ word: String) -> Bool {
        let rootTemp = find(s: word)
        if rootTemp != nil && rootTemp?.isword == true{
            return true
        }
        return false
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    func startsWith(_ prefix: String) -> Bool {
        if find(s: prefix) == nil{
            return false
        }
        return true
    }
    
}

*/

class StockSpanner {
    private var listPrices : [Int]!
    private var listSpans : [Int]!

    init() {
        listPrices = [Int]()
        listSpans = [Int]()
    }
    
    func next(_ price: Int) -> Int {
        guard listPrices.count > 0 else{
            listPrices.append(price)
            listSpans.append(1)
            return 1
        }
        var span : Int = 1
        var index : Int = listPrices.count - 1
        while index >= 0{
            let considerPrice = listPrices[index]
            let considerSpan = listSpans[index]
            if considerPrice > price{
                break
            }
            span += considerSpan
            index -= considerSpan
        }
        listPrices.append(price)
        listSpans.append(span)
        return span
    }
}

