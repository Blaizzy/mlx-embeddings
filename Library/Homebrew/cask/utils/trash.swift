#!/usr/bin/swift

import Foundation

if (CommandLine.arguments.count < 2) {
  exit(2)
}

let manager: FileManager = FileManager()

for item in CommandLine.arguments[1...] {
  do {
    let path: URL = URL(fileURLWithPath: item)
    try manager.trashItem(at: path, resultingItemURL: nil)
    print(path)
  }
  catch {
    print("\0")
  }
}

exit(0)
