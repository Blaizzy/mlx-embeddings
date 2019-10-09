#!/usr/bin/swift

import Foundation

struct swifterr: TextOutputStream {
  public static var stream = swifterr()
  mutating func write(_ string: String) { fputs(string, stderr) }
}

if (CommandLine.arguments.count < 2) {
  exit(2)
}

let manager: FileManager = FileManager()

for item in CommandLine.arguments[1...] {
  do {
    let path: URL = URL(fileURLWithPath: item)
    try manager.trashItem(at: path, resultingItemURL: nil)
    print(path, terminator: "\0")
  }
  catch {
    print(error.localizedDescription, to: &swifterr.stream)
    exit(1)
  }
}

exit(0)
