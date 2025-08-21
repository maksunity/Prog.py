Here is a simple implementation of Bubble Sort in C++. This version includes detailed comments to help understand each step:

```cpp
#include <iostream>
using namespace std;

// Function to perform Bubble Sort on an array
void bubbleSort(int arr[], int n) {
    // Traverse through all array elements
    for (int i = 0; i < n-1; i++) {     
        // Last i elements are already in place, so we don't need to check them again
        for (int j = 0; j < n-i-1; j++) {
            // Traverse the array from 0 to n-i-1
            // Swap if the element found is greater than the next element
            if (arr[j] > arr[j+1]) {
                // Swapping elements
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

// Function to print an array
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Main function to test the Bubble Sort algorithm
int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr)/sizeof(arr[0]);
    
    cout << "Unsorted array: \n";
    printArray(arr, n);
    
    bubbleSort(arr, n);
    
    cout << "\nSorted array: \n";
    printArray(arr, n);
    
    return 0;
}
```

### Explanation:

1. **bubbleSort Function**:
   - Takes an array `arr` and its size `n`.
   - Uses two nested loops to iterate through the array.
     - The outer loop runs `n-1` times because after each complete pass, the largest unsorted element is moved to its correct position.
     - The inner loop compares adjacent elements and swaps them if they are in the wrong order.

2. **printArray Function**:
   - Takes an array `arr` and its size `size`.
   - Iterates through the array and prints each element separated by a space.

3. **main Function**:
   - Initializes an example array.
   - Prints the unsorted array.
   - Calls `bubbleSort` to sort the array.
   - Prints the sorted array.

### Bubble Sort Complexity:
- **Time Complexity**: O(n^2) in the worst and average case, where n is the number of items being sorted. The best-case time complexity is O(n) when the input array is already sorted.
- **Space Complexity**: O(1), as it sorts the array in place without requiring any extra space.

This implementation is straightforward and easy to understand, making it a good choice for educational purposes or small-scale applications where simplicity is more important than performance.