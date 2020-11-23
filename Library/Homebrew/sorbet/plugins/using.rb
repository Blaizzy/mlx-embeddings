# typed: strict
# frozen_string_literal: true

source = ARGV[5]

case source[/\Ausing\s+(.*)\Z/, 1]
when "Magic"
  puts <<-RUBY
    # typed: strict

    class ::Pathname
      sig { returns(String) }
      def magic_number; end

      sig { returns(String) }
      def file_type; end

      sig { returns(T::Array[String]) }
      def zipinfo; end
    end
  RUBY
when "HashValidator"
  puts <<-RUBY
    # typed: strict

    class ::Hash
      sig { params(valid_keys: T.untyped).void }
      def assert_valid_keys!(*valid_keys); end
    end
  RUBY
end
