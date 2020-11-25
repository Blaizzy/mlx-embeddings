# typed: strict
# frozen_string_literal: true

source = ARGV[5]

match = source.match(/\s*def_delegator\s+.*:(?<method>[^:]+)\s*\Z/m)

raise if match.nil?

method = match[:method]

puts <<~RUBY
  # typed: strict

  sig {params(arg0: T.untyped, blk: T.untyped).returns(T.untyped)}
  def #{method}(*arg0, &blk); end
RUBY
