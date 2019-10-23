# frozen_string_literal: true

require "cask/artifact/symlinked"

module Cask
  module Artifact
    class Manpage < Symlinked
      def self.from_args(cask, source)
        section = source.split(".").last

        raise CaskInvalidError, "section should be a positive number" unless section.to_i.positive?

        new(cask, source)
      end

      def initialize(cask, source)
        super
      end

      def resolve_target(_target)
        config.manpagedir.join("man#{section}", target_name)
      end

      def section
        @source.extname.downcase[1..-1].to_s.to_i
      end

      def target_name
        "#{@source.basename(@source.extname)}.#{section}"
      end
    end
  end
end
