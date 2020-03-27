# frozen_string_literal: true

module Utils
  module Shebang
    module_function

    class RewriteInfo
      attr_reader :regex, :max_length, :replacement

      def initialize(regex, max_length, replacement)
        @regex = regex
        @max_length = max_length
        @replacement = replacement
      end
    end

    def rewrite_shebang(rewrite_info, *paths)
      paths.each do |f|
        f = Pathname(f)
        next unless f.file?
        next unless rewrite_info.regex.match?(f.read(rewrite_info.max_length))

        Utils::Inreplace.inreplace f.to_s, rewrite_info.regex, "#!#{rewrite_info.replacement}"
      end
    end
  end
end
