# typed: true
# frozen_string_literal: true

module Language
  # Helper functions for Perl formulae.
  #
  # @api public
  module Perl
    # Helper module for replacing `perl` shebangs.
    module Shebang
      module_function

      def detected_perl_shebang(formula = self)
        perl_path = if formula.uses_from_macos_elements&.include? "perl"
          "/usr/bin/perl#{MacOS.preferred_perl_version}"
        elsif formula.deps.map(&:name).include? "perl"
          Formula["perl"].opt_bin/"perl"
        else
          raise ShebangDetectionError.new("Perl", "formula does not depend on Perl")
        end

        Utils::Shebang::RewriteInfo.new(
          %r{^#! ?/usr/bin/(?:env )?perl( |$)},
          21, # the length of "#! /usr/bin/env perl "
          "#{perl_path}\\1",
        )
      end
    end
  end
end
