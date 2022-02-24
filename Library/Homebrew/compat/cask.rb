# typed: true
# frozen_string_literal: true

module Cask
  class Cask
    extend Enumerable

    def self.each(&block)
      odeprecated "`Enumerable` methods on `Cask::Cask`",
                  "`Cask::Cask.all` (but avoid looping over all casks, it's slow and insecure)"

      return to_enum unless block

      Tap.flat_map(&:cask_files).each do |f|
        yield CaskLoader::FromTapPathLoader.new(f).load(config: nil)
      rescue CaskUnreadableError => e
        opoo e.message
      end
    end
  end
end
