# typed: strict
# frozen_string_literal: true

source = ARGV[5]

source.scan(/:([^?]+\?)/).flatten.each do |method|
  puts <<~RUBY
    # typed: strict

    sig { returns(T::Boolean) }
    def #{method}; end
  RUBY
end
