# typed: strict
# frozen_string_literal: true

source = ARGV[5]

methods = if (single = source[/delegate\s+([^:]+):\s+/, 1])
  [single]
else
  multiple = source[/delegate\s+\[(.*?)\]\s+=>\s+/m, 1]
  non_comments = multiple.gsub(/\#.*$/, "")
  non_comments.scan(/:([^:,\s]+)/).flatten
end

methods.each do |method|
  puts <<~RUBY
    # typed: strict

    sig {params(arg0: T.untyped).returns(T.untyped)}
    def #{method}(*arg0); end
  RUBY
end
