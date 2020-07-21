# frozen_string_literal: true

require "rubocops/extend/formula"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop audits `uses_from_macos` dependencies in formulae
      class UsesFromMacos < FormulaCop
        # Generate with:
        # brew ruby -e 'puts Formula.select {|f| f.keg_only_reason&.provided_by_macos? }.map(&:name).sort.join("\n")'
        # Not done at runtime as its too slow and RuboCop doesn't have access.
        PROVIDED_BY_MACOS_FORMULAE = %w[
          apr
          bc
          bison
          bzip2
          cups
          curl
          dyld-headers
          ed
          expat
          file-formula
          flex
          gcore
          gnu-getopt
          icu4c
          krb5
          libarchive
          libedit
          libffi
          libiconv
          libpcap
          libressl
          libxml2
          libxslt
          llvm
          lsof
          m4
          ncompress
          ncurses
          net-snmp
          openldap
          openlibm
          pod2man
          rpcgen
          ruby
          sqlite
          ssh-copy-id
          swift
          tcl-tk
          texinfo
          unifdef
          unzip
          zip
          zlib
        ].freeze

        # These formulae aren't keg_only :provided_by_macos but are provided by
        # macOS (or very similarly e.g. OpenSSL where system provides LibreSSL)
        # TODO: consider making some of these keg-only.
        ALLOWED_USES_FROM_MACOS_DEPS = (PROVIDED_BY_MACOS_FORMULAE + %w[
          bash
          cpio
          expect
          groff
          gzip
          openssl
          openssl@1.1
          perl
          php
          python
          python@3
          rsync
          vim
          xz
          zsh
        ]).freeze

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          find_method_with_args(body_node, :uses_from_macos, /^"(.+)"/).each do |method|
            dep = if parameters(method).first.class == RuboCop::AST::StrNode
              parameters(method).first
            elsif parameters(method).first.class == RuboCop::AST::HashNode
              parameters(method).first.keys.first
            end

            next if ALLOWED_USES_FROM_MACOS_DEPS.include?(string_content(dep))

            problem "`uses_from_macos` should only be used for macOS dependencies, not #{string_content(dep)}."
          end
        end
      end
    end
  end
end
