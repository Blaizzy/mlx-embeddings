# typed: strict
# frozen_string_literal: true

source = ARGV[5]

source.scan(/:([^\s,]+)/).flatten.each do |method|
  puts <<~RUBY
    # typed: strict

    sig { params(arg: T.untyped).returns(T.untyped) }
    def #{method}(arg = T.unsafe(nil)); end
  RUBY
end
