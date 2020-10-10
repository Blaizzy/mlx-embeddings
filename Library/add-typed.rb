#!/usr/bin/env ruby

require 'pathname'
require 'open3'


Dir.chdir "#{__dir__}/Homebrew"

files = Pathname.glob("**/*.rb").reject { |path| path.to_s.start_with?("vendor/") }

files.each do |file|

  content = file.read

  if content.start_with?("# typed: ")
    puts "Already typed: #{file}"
    next
  end

  ['strict', 'true', 'false', 'ignore'].each do |level|
    puts "Trying #{file} with level #{level}."
    file.write "# typed: #{level}\n#{content.strip}\n"

    output, status = Open3.capture2e('brew', 'typecheck')
    break if status.success?
  end
end
