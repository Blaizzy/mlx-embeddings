# frozen_string_literal: true

module Cask
  class Cmd
    class Outdated < AbstractCommand
      option "--greedy", :greedy, false
      option "--quiet",  :quiet, false
      option "--json",   :json, false

      def initialize(*)
        super
        self.verbose = ($stdout.tty? || verbose?) && !quiet?
        @outdated_casks = casks(alternative: -> { Caskroom.casks }).select do |cask|
          odebug "Checking update info of Cask #{cask}"
          cask.outdated?(greedy?)
        end
      end

      def run
        output = @outdated_casks.map { |cask| cask.outdated_info(greedy?, verbose?, json?) }

        puts json? ? JSON.generate(output) : output
      end

      def self.help
        "list the outdated installed Casks"
      end
    end
  end
end
