# typed: strict
# frozen_string_literal: true

source = ARGV[5]

/\busing +Magic\b/.match(source) do |_|
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
end
