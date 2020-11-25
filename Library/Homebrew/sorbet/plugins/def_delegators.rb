# typed: strict
# frozen_string_literal: true

source = ARGV[5]

symbols = source.scan(/:[^\s,]+/)

_, *methods = symbols.map { |s| s.delete_prefix(":") }

methods.each do |method|
  puts <<~RUBY
    # typed: strict

    sig {params(arg0: T.untyped, blk: T.untyped).returns(T.untyped)}
    def #{method}(*arg0, &blk); end
  RUBY
end
